import torch
import torch.nn as nn

from models.common import RevIN
from models.common import DDI
from models.common import MDM
from models.tsmoe import AMS


class Model(nn.Module):
    """Implementation of AMD."""

    def __init__(self, configs):
        super(Model, self).__init__()

        # input_shape[batch_size, feature_num, seq_len]
        # pred_len, {96, 192,336,720}
        # n_block：DDI层数, 全是1
        # dropout,0.2
        # patch, 变化的
        # k：MDM层数，下采样的层数，取3
        # c:downsampling rate ， 取2
        # alpha， 变化的
        # target_slice，原文中明确指出，AMD 模型默认采用 Channel Independence (CI) 策略，即每个通道独立预测，target_slice=None 或不设置
        # norm=True, layernorm=True
        self.target_slice = None
        self.norm = configs.norm


        if self.norm:
            self.rev_norm = RevIN(configs.input_shape[-1])

        self.pastmixing = MDM(configs.input_shape, k=configs.k, c=configs.c, layernorm=configs.layernorm)

        self.fc_blocks = nn.ModuleList([DDI(configs.input_shape, dropout=configs.dropout, patch=configs.patch, alpha=configs.alpha, layernorm=configs.layernorm)
                                        for _ in range(configs.n_block)])

        self.moe = AMS(configs.input_shape, configs.pred_len, ff_dim=2048, dropout=configs.dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]
        # 添加维度保护逻辑
        original_dim = x.dim()
        if original_dim == 2:  # 单变量模式 [batch, seq_len]
            x = x.unsqueeze(-1)  # [32,96] -> [32,96,1]
        # layer norm

        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)

        # [batch_size, feature_num, seq_len]
        time_embedding = self.pastmixing(x)

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # MOE
        x, moe_loss = self.moe(x, time_embedding)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            x = x[:, :, self.target_slice]
            print("fff", x.shape)

        # 恢复原始维度
        if original_dim == 2:
            x = x.squeeze(-1)  # 压缩多余特征维度
        return x, moe_loss