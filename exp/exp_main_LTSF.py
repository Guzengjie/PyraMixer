from sympy.codegen import Print
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (FEDformer, PatchTST, DLinear, TimesNet, SCINet, LightTS, \
                    PyraMixer, AMD, TimeMixer)
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import detrend


warnings.filterwarnings('ignore')


@torch.no_grad()
def estimate_snr(residuals_np: np.ndarray) -> float:
    """
    简单信噪比估计：把高频段 (>75% 频点) 视为噪声，其余视为信号
    residuals_np: shape (N*L,)
    return: SNR (dB)
    """
    f, Pxx = welch(residuals_np, nperseg=min(1024, len(residuals_np)))
    idx_high = f > np.percentile(f, 75)          # 高频
    P_signal = Pxx[~idx_high].sum()
    P_noise  = Pxx[idx_high].sum()
    snr_db = 10 * np.log10(P_signal / (P_noise + 1e-12))
    return snr_db

@torch.no_grad()
def compute_spectral_entropy(residuals, fs=24, nperseg=None):
    x = detrend(residuals - np.mean(residuals))  # 去线性趋势
    x = (x - x.mean()) / (x.std() + 1e-12)  # z-score
    _, Pxx = welch(x, fs,nperseg=min(256, len(x)))  # 保证分辨率
    P_norm = Pxx / Pxx.sum()
    H = -np.sum(P_norm * np.log(P_norm + 1e-12))
    return float(H)

@torch.no_grad()
def compute_alpha_once(model, train_loader, device, pred_len, f_dim, beta=1.0):
    model.eval()
    all_r = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if i < 10000:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                pred = model(batch_x)
                batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
                r = (batch_y - pred).detach().cpu().numpy()
                all_r.append(r)
            else:
                break
        all_r = np.concatenate(all_r, axis=0)
        print(all_r.shape)

        snr_db = estimate_snr(all_r.reshape(-1))
        hl = []
        for h in range(7):
            y = all_r[:, :, h]
            y = y.reshape(-1)
            H = compute_spectral_entropy(y)
            hl.append(H)
        print(np.mean(hl))
        # 将 SNR(dB) 线性映射到 0~1 作为 α
        alpha = torch.sigmoid(torch.tensor(snr_db / 10.0)).item()
        print(alpha)
        alpha = float(np.clip(alpha * beta, 0.01, 0.99))
        print(f"[SNR] train SNR = {snr_db:.2f} dB → α = {alpha:.3f}")
        print("h", H)
        return alpha


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'SCINet': SCINet,
            'LightTS': LightTS,
            'PyraMixer': PyraMixer,
            'AMD': AMD,
            'TimeMixer': TimeMixer,

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model_type == 'former' or self.args.model == 'TimesNet' or self.args.model == 'LightTS' or self.args.model == 'TimeMixer'or self.args.model == 'MICN'or self.args.model == 'SCINet':
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.model == 'AMD':
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # f_dim = -1 if self.args.features == 'MS' else 0
        # alpha = compute_alpha_once(self.model, train_loader, self.device, self.args.pred_len, f_dim, beta=1.0)
        # self.model.freq_gate._alpha.fill_(alpha)  # 直接写进模块的 buffer
        # self.model.train()  # 恢复训练模式

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        path = self.args.checkpoints + self.args.model + '/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        final_train_loss = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.model_type == 'former' or self.args.model == 'TimesNet' or self.args.model == 'LightTS' or self.args.model == 'TimeMixer'or self.args.model == 'MICN' or self.args.model == 'SCINet':
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # print(batch_x.shape, batch_x_mark.shape, dec_inp.shape, batch_y_mark.shape)
                    elif self.args.model == 'AMD':
                        # 把整个一步都包起来
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 10 == 0:
                # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                # speed = (time.time() - time_now) / iter_count
                # left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                # iter_count = 0
                # time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time), len(train_loader), i)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            early_stopping(train_loss, vali_loss, self.model, path)
            final_train_loss = early_stopping.c_train_loss
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 早停
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print(self.args.model , "模型文件在： ", best_model_path)

        return self.model, final_train_loss

    def check_train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.args.stat_para:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # from ptflops import get_model_complexity_info
            # fl, pa = get_model_complexity_info(self.model, (self.args.seq_len, self.args.enc_in), as_strings=True,print_per_layer_stat=False)
            # print("flops", fl)
            # print("参数量", pa)

        time_now = time.time()

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        path = self.args.checkpoints + self.args.model + '/' + 'checkpoint_train_' + setting
        start_epoch = -1
        counter = 0
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            path_checkpoint = path + '/' + 'checkpoint_' + str(self.args.checkpoint_epoch_best) + '.pth'  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            model_optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = self.args.checkpoint_epoch_start  # 设置开始的epoch
            counter = self.args.checkpoint_counter_start

            for param_group in model_optim.param_groups:
                param_group['lr'] = checkpoint['lr']
            adjust_learning_rate(model_optim, start_epoch + 1, self.args)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, counter=counter)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch + 1, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model_type == 'former' or self.args.model == 'TimesNet' or self.args.model == 'LightTS' or self.args.model == 'TimeMixer'or self.args.model == 'MICN'or self.args.model == 'SCINet':
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.model == 'AMD':
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time), len(train_loader), i)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            early_stopping(vali_loss, self.model, path, self.args.checkpoint_train, model_optim, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 早停
        lr = 0
        for param_group in model_optim.param_groups:
            lr = param_group['lr']
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': model_optim.state_dict(),
            "epoch": self.args.train_epochs,
            "lr": lr,
        }
        self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        task_flag = 'test'
        if test == 1:
            test_data, test_loader = self._get_data(flag='val')
            task_flag = 'val'
            print(">>" * 40, "val_metrics")
        else:
            print(">>" * 40, "test_metrics")

        if self.args.checkpoint_train:
            print('loading model')
            path = self.args.checkpoints + self.args.model + '/' + 'checkpoint_train_' + setting
            path_checkpoint = path + '/' + 'checkpoint_' + str(self.args.checkpoint_epoch_best) + '.pth'  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        preds = []
        trues = []
        self.model.eval()
        # summary(self.model, input_size=(1, self.args.seq_len, self.args.enc_in))
        # if test == 1:
        macs, params, avg_time = self.other_metrics(self.model, self.args.model, self.args.seq_len, self.args.pred_len, self.args.enc_in)

        with torch.no_grad():
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model_type == 'former' or self.args.model == 'TimesNet' or self.args.model == 'LightTS' or self.args.model == 'TimeMixer'or self.args.model == 'MICN'or self.args.model == 'SCINet':
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.model == 'AMD':
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                if self.args.checkpoint_train:
                    print("cost time: {}".format(time.time() - epoch_time), len(test_loader), i)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + self.args.model + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        if test == 0:
            print('mse:{}, mae:{}'.format(mse, mae))
        # elif test == 1:
        #     print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(task_flag + '__' + setting + "  \n")
        f.write('mse:{}, mae:{}'.format('%.4f' % mse, '%.4f' % mae))
        f.write('\n')
        f.write('\n')
        f.close()

        metrics = {
            'mae': np.asarray(mae),
            'mse': np.asarray(mse),
            'rmse': np.asarray(rmse),
            'mape': np.asarray(mape),
            'corr': np.asarray(corr)
        }
        np.save(folder_path + task_flag + 'metrics.npy', metrics, allow_pickle=True)
        # np.save(folder_path + task_flag + 'metrics.npy', np.array([mae, mse, rmse, mape, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse, mae, macs, params, avg_time

    def predict(self, setting, load=False):
        test_data, pred_loader = self._get_data(flag='test')
        path = self.args.checkpoints + self.args.model + '/' + setting


        if load:
            # path = os.path.join(self.args.checkpoints, setting)
            path = self.args.checkpoints + self.args.model + '/' + setting
            best_model_path = path + '/' + 'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model_type == 'former' or self.args.model == 'TimesNet' or self.args.model == 'LightTS' or self.args.model == 'TimeMixer' or self.args.model == 'MICN' or self.args.model == 'SCINet':
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    elif self.args.model == 'AMD':
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + 'real_prediction/'+ self.args.model + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mse_f = []
        for i in range(preds.shape[2]):
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds[:, :, i], trues[:, :,i])
            mse_f.append(mse)

        print("ggg", np.argsort(mse_f))

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        print('mse:{}, mae:{}'.format(mse, mae))


        # result save
        print(preds.shape)
        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_trues.npy', trues)

        print(folder_path + 'real_prediction.npy')

        return

    def other_metrics(self, model, model_name, seq_len, pre_len, enc_in):
        # ETT:4, Weather:5
        if self.args.model == 'TimeMixer' or self.args.model == 'TimesNet'or self.args.model == 'LightTS'or self.args.model == 'SCINet':
            b_x = torch.randn(1, seq_len, enc_in).cuda()
            b_xm = torch.randn(1, seq_len, self.args.freq_dim).cuda()
            dec_in = torch.randn(1, seq_len+pre_len, enc_in).cuda()
            b_ym = torch.randn(1, seq_len+pre_len, self.args.freq_dim).cuda()
            macs, params = profile(model, inputs=(b_x,b_xm, dec_in, b_ym, ))

            print(f"MACs: {macs / 1e9:.4f} G")
            print(f"参数量: {params / 1e6:.2f} M")
            torch.cuda.reset_peak_memory_stats()  # 清空历史峰值
            with torch.no_grad():
                _ = self.model(b_x,b_xm, dec_in, b_ym)
            peak = torch.cuda.max_memory_allocated()  # Byte → MB
            print(f'峰值显存: {peak / 1024 ** 2:.1f} MB')

            with torch.no_grad():
                for _ in range(10):  # warmup
                    _ = model(b_x,b_xm, dec_in, b_ym)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(100):
                    _ = model(b_x,b_xm, dec_in, b_ym)
                torch.cuda.synchronize()
                avg_time = (time.time() - start) / 100

            print(f"平均推理时间: {avg_time * 1000:.2f} ms")
            return macs / 1e9, params / 1e6, avg_time * 1000

        else:
            dummy = torch.randn(1, seq_len, enc_in).cuda()
            macs, params = profile(model, inputs=(dummy, ))

            print(f"MACs: {macs / 1e9:.4f} G")
            print(f"参数量: {params / 1e6:.2f} M")
            # 1. 参数量
            # params = sum(p.numel() for p in self.model.parameters())
            # train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            # print(f'总参数量: {params / 1e6:.2f} M, 可训练: {train_params / 1e6:.2f} M')
            # 2. 显存占用（包括参数 + 中间激活）
            torch.cuda.reset_peak_memory_stats()  # 清空历史峰值
            with torch.no_grad():
                _ = self.model(dummy)
            peak = torch.cuda.max_memory_allocated()  # Byte → MB
            print(f'峰值显存: {peak / 1024 ** 2:.1f} MB')

            with torch.no_grad():
                for _ in range(10):  # warmup
                    _ = model(dummy)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(100):
                    _ = model(dummy)
                torch.cuda.synchronize()
                avg_time = (time.time() - start) / 100

            print(f"平均推理时间: {avg_time * 1000:.2f} ms")
            return macs / 1e9, params / 1e6, avg_time * 1000

