import argparse
from exp.exp_main_LTSF import Exp_Main
import random
import numpy as np
import xlrd
from xlutils.copy import copy
import torch

fix_seed = [2024, 2022, 2023, 2025, 2026]

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
seq_len = 96
pre_len = 96
epoch = 10
batch = 32
modelname = 'AMD'
modeltype = 'for'  # 'former', 'TimeMixer', 'TimeNet'
dataset_name = 'ETTh1'


parser.add_argument('--stat_para', type=int, default=0)
parser.add_argument('--checkpoint_train', type=int, default=0)  # 0  是否要启用断点续训
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

# AMD
# model define
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
parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')  # 48
parser.add_argument('--pred_len', type=int, default=pre_len, help='prediction sequence length')  # 96

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
# input_shape = [batch, feature_num, seq_len]
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--input_shape', type=list, default=[96, 7], help='[seq_len, enc_in]')
parser.add_argument('--n_block', type=int, default=1)
parser.add_argument('--patch', type=int, default=4, help='patch length')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--c', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--norm', type=bool, default=True)
parser.add_argument('--layernorm', type=bool, default=True)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')




args = parser.parse_args()

data_parser = {
                'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'norm':[1], 'ln':[1], 'freq':['h'], 'patch':[16], 'layernorm':[True]},
                'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'norm':[1], 'ln':[1], 'freq':['h'], 'patch':[4], 'layernorm':[False]},
                'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[2, 4, 8, 16], 'norm':[1], 'ln':[1], 'freq':['t'], 'patch':[16], 'layernorm':[True]},
                'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'norm':[1], 'ln':[0], 'freq':['t'], 'patch':[8], 'layernorm':[True]},
                'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1], 'idl': [0], 'hfs':[8, 4, 4, 4], 'norm':[0], 'ln':[1], 'snet_act':[0], 'patch':[1], 'layernorm':[True]},
                'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1], 'idl': [0], 'hfs':[12, 6, 4, 4], 'norm':[1], 'ln':[1], 'freq':['h'], 'patch':[16], 'layernorm':[False]},
                'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1], 'idl': [0], 'ln':[1]},
                'ER': {'data': 'ER.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1], 'idl': [0], 'hfs':[8, 4, 4, 4], 'norm':[1], 'ln':[0], 'freq':['d'], 'patch':[4], 'layernorm':[True]},
                'Weather': {'data': 'Weather.csv', 'T': 'CO2 (ppm)', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1], 'idl': [0], 'hfs': [8, 4, 4, 4], 'norm': [1], 'ln': [1], 'freq':['t'], 'patch':[16], 'layernorm':[True]},
                'ILI': {'data': 'ILI.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                'idl': [0], 'hfs': [48, 24, 16, 16], 'norm': [1], 'ln': [0], 'freq':['7d'], 'patch':[4], 'layernorm':[True]},
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
                args.patch = data_info['patch'][0]
                args.freq = data_info['freq'][0]
                args.layernorm = data_info['layernorm'][0]

            args.input_shape = [args.seq_len, args.enc_in]

            if args.is_training:
                idx = 0
                best_val_mse = 100
                best_test_mse = 100
                best_test_mae = 100
                for ii in range(exp_num):
                    random.seed(fix_seed[ii])
                    np.random.seed(fix_seed[ii])
                    torch.manual_seed(fix_seed[ii])
                    setting = '{}_{}_{}_sl{}_pl{}_b{}_p{}_k{}_c{}_alpha{}_norm{}_seed{}_itr{}'.format(
                        args.model,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.pred_len,
                        args.batch_size,
                        args.patch,
                        args.k,
                        args.c,
                        args.alpha,
                        args.norm,
                        args.layernorm,
                        fix_seed[ii],ii)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}'.format(setting))
                    exp.train(setting)
                    print('>>>>>>>testing : {}'.format(setting))
                    mse, mae, macs, params, avg_time = exp.test(setting)
                    val_mse, val_mae, _, _, _ = exp.test(setting, 1)

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
                setting = '{}_{}_{}_sl{}_pl{}_b{}_p{}_k{}_c{}_alpha{}_norm{}_seed{}_itr{}'.format(
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.batch_size,
                    args.patch,
                    args.k,
                    args.c,
                    args.alpha,
                    args.norm,
                    args.layernorm,
                    fix_seed[ii], ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting,load=1)   # 0是输入测试集数据，1是输入验证集数据
                # exp.test(setting)

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


