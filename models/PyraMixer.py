import torch
import torch.nn as nn
import numpy as np
import torch.fft as fft
from torch.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


cuda = True if torch.cuda.is_available() else False


class Splitting(nn.Module):
    def __init__(self, num):
        super(Splitting, self).__init__()
        self.num = num

    @staticmethod
    def even(x):
        if type(x) == list:
            x1 = x.copy()
            x2 = x1[0]
            return x2[:, ::2, :]
        return x[:, ::2, :]

    @staticmethod
    def odd(x):
        if type(x) == list:
            x1 = x.copy()
            x2 = x1[1]
            return x2[:, 1::2, :]
        return x[:, 1::2, :]

    @staticmethod
    def zip_up_the_pants(even, odd):
        even = even.permute(1, 0, 2)
        odd1 = odd[0]
        odd2 = odd[1]
        odd1 = odd1.permute(1, 0, 2)  # L, B, D
        return torch.cat((even, odd1), 0).permute(1, 0, 2), odd2  # B, L, D

    def forward(self, x):
        x = x.permute(1, 0, 2)
        idx = torch.arange(x.shape[0])
        idx1 = idx[::self.num]  # 24-8-3,21-7-3, 18-6-3, 15-5-3, 12-4-3, 9-3-3, 6-2-3
        idx2 = np.delete(idx, idx1)
        x1 = x[idx2]
        x1 = x1.permute(1, 0, 2)
        x2 = x[::self.num]
        x2 = x2.permute(1, 0, 2)

        return x1, x2


class DGS(nn.Module):

    def __init__(self, num):
        super(DGS, self).__init__()
        self.split_list = []
        for i in range(2, num + 1):
            self.split_list.append(Splitting(i))
        self.num = num

    def forward(self, x):
        y = x
        dian = {}
        for i in range(self.num - 1):
            (remain, x) = self.split_list[self.num - 2 - i](y)
            y = remain
            dian[self.num - i] = x
            dian['rest' + '{}'.format(self.num - i)] = remain
        return dian

class FFTMix(nn.Module):
    def __init__(self, enc_in):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, enc_in))

    def forward(self, x1, x2):
        f1 = torch.fft.rfft(x1, dim=1)
        f2 = torch.fft.rfft(x2, dim=1)
        f = self.w * f1 + (1 - self.w) * f2
        return torch.fft.irfft(f, n=x1.shape[1], dim=1)


class SampleLinear(nn.Module):
    def __init__(self, ln, snet_act, min_len, c_level, k_of_pyramid, deep_max, enc_in, fft):
        super(SampleLinear, self).__init__()
        self.c_level = c_level
        current_mlp_size1 = min_len + (c_level - 2) * min_len
        self.current_mlp_size2 = current_mlp_size1*1
        self.fft_ornot = fft
        if not fft:
            self.current_mlp_size2 = current_mlp_size1*2
        linear_size = min_len + (c_level+k_of_pyramid - 3) * min_len  # 转换到（k+2）层的下一层（k+1）,可以对齐维度
        self.over2 = nn.Linear(current_mlp_size1, linear_size)
        self.over3 = nn.Linear(self.current_mlp_size2, linear_size)
        self.over4 = nn.Linear(self.current_mlp_size2, current_mlp_size1)
        self.norm = nn.LayerNorm(linear_size)
        self.norm_final = nn.LayerNorm(current_mlp_size1)
        self.LN = ln
        self.b_act = snet_act
        self.act = nn.GELU()
        self.deep_max = deep_max
        self.fft = FFTMix(enc_in)


    def forward(self, x):
        if not type(x) == list:
            x2 = x['rest' + '{}'.format(self.c_level)]
            x2 = x2.permute(0, 2, 1)
            x2 = self.over2(x2)

            if self.LN:
                x2 = self.norm(x2)

            if self.b_act:
                x2 = self.act(x2)

            x2 = torch.cat((x[self.c_level], x2.permute(0, 2, 1)), dim=1)

            return x2, x
        else:
            a = x[1]
            x2 = a['rest' + '{}'.format(self.c_level)]

            # 前
            if self.fft_ornot:
                x2 = self.fft(x2, x[0])
            else:
                x2 = torch.cat((x2, x[0]), dim=1)

            x2 = x2.permute(0, 2, 1)
            if self.c_level == self.deep_max:
                x2 = self.over4(x2)
                if self.LN:
                    x2 = self.norm_final(x2)
            else:
                x2 = self.over3(x2)
                if self.LN:
                    x2 = self.norm(x2)

            if self.b_act:
                x2 = self.act(x2)

            x2 = torch.cat((a[self.c_level], x2.permute(0, 2, 1)), dim=1)

            return x2, a


class MSM(nn.Module):

    def __init__(self, ln, snet_act, deepest_layer, min_len, c_level, k_of_pyramid, p_of_pyramid, enc_in, fft):
        super(MSM, self).__init__()
        self.c_level = c_level
        self.deepest_layer = deepest_layer
        self.circulate_linear = SampleLinear(ln, snet_act, min_len, self.c_level, k_of_pyramid, self.deepest_layer, enc_in, fft)
        if self.c_level < deepest_layer:
            self.encoder_block = MSM(ln, snet_act, deepest_layer, min_len, self.c_level + k_of_pyramid, k_of_pyramid, p_of_pyramid, enc_in, fft)


    def forward(self, x):
        x1, list1 = self.circulate_linear(x)
        new_data = [x1, list1]
        if self.c_level == self.deepest_layer:  # 最深层(32, 2, 60)
            aa = x1  # 连接输出
            return aa
        else:
            # 下一层的二叉树
            return self.encoder_block(new_data)  # ------------3个数据作为上输入


class LRSNU(nn.Module):
    def __init__(self, ln, snet_act, seq_len, min_len, k_of_pyramid, p_of_pyramid, enc_in, fft):
        super(LRSNU, self).__init__()
        h_f = int(min_len * (1+k_of_pyramid*p_of_pyramid))
        self.earlier_projection = nn.Linear(seq_len, h_f)
        self.later_projection = nn.Linear(h_f, 512)  # 512
        deepest_layer = int((1+k_of_pyramid*p_of_pyramid))
        self.aggregator = MSM(ln, snet_act, deepest_layer, min_len, k_of_pyramid+1, k_of_pyramid, p_of_pyramid, enc_in, fft)
        self.data_pre = DGS(deepest_layer)
        self.linear = nn.Linear(512, seq_len)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.earlier_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.data_pre(x)
        x = self.aggregator(x)  # 从最低下开始
        x = self.later_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act(x)
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.norm = configs.Normalization
        self.arg = configs.ar
        self.pre_len = configs.pred_len
        self.block1 = LRSNU(configs.LN, configs.snet_act, configs.seq_len, configs.subseq_len, configs.k_of_pyramid[0], configs.p_of_pyramid[0],  configs.enc_in, configs.fft)
        self.decoder = nn.Linear(configs.seq_len, configs.pred_len)
        self.before = nn.Linear(configs.seq_len, configs.seq_len)
        self.act = nn.ReLU()
        self.scale = max(0, (1 - configs.seq_len / 336)*configs.alpha*(1 - configs.pred_len / 1024))
        self.adf = nn.Parameter(configs.adf)

    def forward(self, x):
        y = x
        if self.norm == 1:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= std

        x = self.block1(x)
        # x = self.linear_out(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.arg:
            h = self.before(y.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.act(h)
            x = x + h*self.adf*self.scale

        x = self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.norm == 1:
            x = x * \
                (std[:, 0, :].unsqueeze(1).repeat(1, self.pre_len, 1))
            x = x + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.pre_len, 1))
        return x
