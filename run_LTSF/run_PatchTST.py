import argparse
from exp.exp_main_LTSF import Exp_Main
import random
import numpy as np
import xlrd
from xlutils.copy import copy
import pandas as pd

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# from torchsummary import summary
fix_seed = [2024, 2022, 2023, 2025, 2026]

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
seq_len = 96
pre_len = 96
epoch = 10
batch = 32
modelname = 'PatchTST'
modeltype = 'for'  # 'former', SNet
dataset_name = 'ETTh1'

# ETT，Normalization=1， LN=1, h_fs=[12, 6, 4, 4], CI=0
# ER. Normalization=0， LN=0, h_fs=[8, 4, 4, 4],  CI=1
# WTH. Normalization=0， LN=1, h_fs=[8, 6, 4, 4],  CI=0
# ECL. Normalization=1， LN=1, h_fs=[8, 4, 4, 4],  CI=0

# SNet set
sub_seq = [48, 24, 12, 6]  # seq //4, seq//8, seq//16

parser.add_argument('--s_individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--stat_para', type=int, default=0)
parser.add_argument('--subseq_len', type=list, default=sub_seq)
parser.add_argument('--Normalization', type=int, default=1, help='Non-stationary Transformer')  # 1
parser.add_argument('--ar', type=int, default=1, help='Non-stationary Transformer')  # 0
parser.add_argument('--adf', type=int, default=[1, 1, 1, 1, 1, 1])  # 0
parser.add_argument('--LN', type=int, default=0, help='LN')  # 1
parser.add_argument('--snet_act', type=int, default=1)  # 0
parser.add_argument('--h_fs', type=int, default={24: 60, 18: 60, 9: 24, 6: 24}, help='max_num_patch')  # 8
parser.add_argument('--checkpoint_train', type=int, default=0)  # 0
parser.add_argument('--checkpoint_epoch_best', type=int, default=4)  # 看pth文件序号
parser.add_argument('--checkpoint_epoch_start', type=int, default=3)  # 已经完成的最大epoch，重启的epoch
parser.add_argument('--checkpoint_counter_start', type=int, default=3)  # 重启的counter


# data loader
datapath = '/home/gzj/长期预测数据集1/'
parser.add_argument('--data', type=str, default=dataset_name, help='dataset type')
parser.add_argument('--root_path', type=str, default=datapath, help='root path of the data file')
parser.add_argument('--data_path', type=str, default=dataset_name + '.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default=modelname,
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')
parser.add_argument('--model_type', type=str, default=modeltype, help='task id')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Wavelets',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')  # 32
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# forecasting task
parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')  # 48
parser.add_argument('--label_len', type=int, default=seq_len, help='start token length')  # 48
parser.add_argument('--pred_len', type=int, default=pre_len, help='prediction sequence length')  # 96
parser.add_argument('--embed_type', type=int, default=0, help='prediction sequence length')
# parser.add_argument('--cross_activation', type=str, default='tanh'

# model define mode_select
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--moving_avg', default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=epoch, help='train epochs')
parser.add_argument('--batch_size', type=int, default=batch, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # 3
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate, type1-4，type3不改学习率')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# 根据原文调整参数
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')  # 4：ILI、ETTH,     16
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--patch_len', type=int, default=24, help='patch length')  # ILI:24
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')  # 128：ILI、ETTH,      256



args = parser.parse_args()

# sum_seq_len = {24:[12, 6, 3], 48:[24, 12, 6, 3], 96:[48, 24, 12, 6], 192:[96, 48, 24, 12], 336:[168, 84, 42, 21], 720:[360, 180, 90, 45]}
# sum_seq_len = {24:[48, 24, 12, 6], 48:[48, 24, 12, 6], 96:[48, 24, 12, 6], 192:[48, 24, 12, 6], 336:[48, 24, 12, 6], 720:[48, 24, 12, 6]}
sum_seq_len = {48: [48, 24, 12, 6], 96: [48, 24, 12, 6], 36: [48, 24, 12, 6], 192: [48, 24, 12, 6], 336: [48, 24, 12, 6], 720: [48, 24, 12, 6]}

def get_seq_num(sum_seq_len, int_len, int_num):
    int_sub_len = sum_seq_len[int_len]
    int_seq_num = {}
    j = 0
    for i in int_sub_len:
        int_seq_num[i] = int_num[j]
        j = 1 + j
    return int_sub_len, int_seq_num
data_parser = {
                'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'n_heads':[4], 'patch_len':[48], 'd_ff':[128], 'freq':['h']},
                'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'n_heads':[4], 'patch_len':[48], 'd_ff':[128], 'freq':['h']},
                'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[2, 4, 8, 16], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256], 'freq':['t']},
                'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256], 'freq':['t']},
                'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1], 'idl': [0], 'hfs':[8, 4, 4, 4], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256]},
                'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256], 'freq':['h']},
                'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1], 'idl': [0], 'ln':[1]},
                'ER': {'data': 'ER.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1], 'idl': [0], 'hfs':[8, 4, 4, 4], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256], 'freq':['d']},
                'Weather': {'data': 'Weather.csv', 'T': 'CO2 (ppm)', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1], 'idl': [0], 'hfs': [8, 4, 4, 4], 'n_heads':[16], 'patch_len':[48], 'd_ff':[256], 'freq':['t']},
                'ILI': {'data': 'ILI.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                'idl': [0], 'hfs': [48, 24, 16, 16], 'n_heads':[4], 'patch_len':[24], 'd_ff':[128], 'freq':['7d']},
}

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# print('Args in experiment:')
# print(args)

if __name__ == '__main__':
    Exp = Exp_Main
    # all_task = {'seq_len': [96], 'pre_len': [96, 192, 336, 720], 'dataset_name': ['ETTh1', 'ETTh2', 'ETTm1', 'ECL', 'Weather', 'ER', 'ILI'],
    #             'freq': ['h', 'h', 't', 'h', 't', 'd', '7d']}
    # ILI, 输入36， 预测[24, 36, 48, 60]
    all_task = {'seq_len': [96], 'pre_len': [96],
                'dataset_name': ['ETTh1']}
    exp_num = 1
    best_all_test_mse = []
    best_all_test_mae = []
    best_all_test_idx = []
    for x1 in range(len(all_task['dataset_name'])):
        for x2 in range(len(all_task['pre_len'])):
            args.seq_len = all_task['seq_len'][0]
            args.label_len = all_task['seq_len'][0]
            args.pred_len = all_task['pre_len'][x2]
            args.data = all_task['dataset_name'][x1]
            pl = []
            if args.data in data_parser.keys():
                data_info = data_parser[args.data]
                args.data_path = data_info['data']
                args.target = data_info['T']
                args.enc_in, args.dec_in, args.c_out = data_info[args.features]
                args.s_individual = data_info['idl'][0]
                args.n_heads = data_info['n_heads'][0]
                args.patch_len = data_info['patch_len'][0]
                args.d_ff = data_info['d_ff'][0]
                args.subseq_len, args.h_fs = get_seq_num(sum_seq_len, args.seq_len, data_info['hfs'])
                args.freq = data_info['freq'][0]

                print(args.subseq_len, args.h_fs)
                if args.data == 'ECL':
                    pl = np.ones(321)
                else:
                    pl = np.ones(data_info['M'][0])

            df_raw = pd.read_csv(args.root_path + args.data_path)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            y = np.array(df_data)

            # for i in range(y.shape[1]):
            #     train = int(y.shape[0] * 0.6)
            #     res = adfuller(y[:train, i])
            #     if res[1] < 0.01 and res[0] < res[4]['1%']:  # ADF检验，稳定序列
            #         pl.append(0.0)
            #         print(y.shape[1], i, 1)
            #     else:  # ADF检验，不稳定序列
            #         pl.append(1.0)
            #         print(y.shape[1], i, 0)

            pl = list(map(float, pl))
            pl = [1-i for i in pl]
            adf = torch.tensor(pl).to('cuda:0')
            adf = adf.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, args.seq_len, 1)
            args.adf = adf

            if args.is_training:
                idx = 0
                best_val_mse = 100
                best_test_mse = 100
                best_test_mae = 100
                for ii in range(exp_num):
                    random.seed(fix_seed[ii])
                    torch.manual_seed(fix_seed[ii])
                    np.random.seed(fix_seed[ii])
                    setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}_{}'.format(
                        args.model,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.revin, fix_seed[ii],ii)
                    # print(args.data, args.model, args.seq_len, args.pred_len, args.batch_size)

                    exp = Exp(args)  # set experiments
                    if args.checkpoint_train:
                        print('>>>>>>>restart checkpoint training : {}'.format(setting))
                        exp.check_train(setting)
                    else:
                        print('>>>>>>>start training : {}'.format(setting))
                        exp.train(setting)
                    print('>>>>>>>testing : {}'.format(setting))
                    mse, mae = exp.test(setting)
                    val_mse, val_mae = exp.test(setting, 1)

                    if val_mse < best_val_mse:
                        idx = ii
                        best_test_mae = mae
                        best_test_mse = mse
                        best_val_mse = val_mse
                        # print(">>"*40, 'mse:', '%.3f' % mse, '  mae:', '%.3f' % mae, ">>"*40, idx)
                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)

                    torch.cuda.empty_cache()
                best_all_test_mse.append('%.3f' % best_test_mse)
                best_all_test_mae.append('%.3f' % best_test_mae)
                best_all_test_idx.append(idx)
                print(">>" * 40, 'mse:', '%.3f' % best_test_mse, '  mae:', '%.3f' % best_test_mae, ">>" * 10, idx)

            else:
                ii = 0
                setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}_{}'.format(
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.revin, fix_seed[ii], ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, load=1)  # 0是输入测试集数据，1是输入验证集数据

                torch.cuda.empty_cache()

    file = r"/home/gzj/former模型/test_results/multi_test_results/multi_test_results.xls"
    oldwb = xlrd.open_workbook(file)  # 打开工作簿
    sheet_names = oldwb.sheet_names()
    newwb = copy(oldwb)
    if modelname not in sheet_names:
        model_sheet = newwb.add_sheet(modelname)
        for i in range(len(best_all_test_mae)):
            model_sheet.write(i, 1, best_all_test_mse[i])
            model_sheet.write(i, 2, best_all_test_mae[i])
            model_sheet.write(i, 3, best_all_test_idx[i])
            newwb.save(file)  # 保存修改
    else:
        model_sheet = newwb.get_sheet(modelname)
        for i in range(len(best_all_test_mae)):
            model_sheet.write(i, 1, best_all_test_mse[i])
            model_sheet.write(i, 2, best_all_test_mae[i])
            model_sheet.write(i, 3, best_all_test_idx[i])
            newwb.save(file)  # 保存修改


