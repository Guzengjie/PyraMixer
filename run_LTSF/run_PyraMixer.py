import argparse
from exp.exp_main_LTSF import Exp_Main
import random
import numpy as np
import xlrd
from xlutils.copy import copy
import pandas as pd

import torch
print(torch.version.cuda)
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
fix_seed = [2024]  # 2024

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
seq_len = 96
pre_len = 96
epoch = 10
batch = 32
modelname = 'PyraMixer'
modeltype = 'for'  # 'former', SNet
dataset_name = 'ETTh1'

sub_seq = [48]

parser.add_argument('--stat_para', type=int, default=0)
parser.add_argument('--subseq_len', type=int, default=seq_len//2, help='下采样的最短长度')
parser.add_argument('--k_of_pyramid', type=list, default=sub_seq, help='划分金字塔的个数')
parser.add_argument('--p_of_pyramid', type=list, default=sub_seq, help='子金字塔的层数')
parser.add_argument('--Normalization', type=int, default=1, help='Non-stationary Transformer')
parser.add_argument('--ar', type=int, default=1, help='Non-stationary Transformer')
parser.add_argument('--adf', type=int, default=[1])
parser.add_argument('--LN', type=int, default=0, help='LN')
parser.add_argument('--snet_act', type=int, default=1)
parser.add_argument('--checkpoint_train', type=int, default=0)
parser.add_argument('--fft', type=bool, default=True)
parser.add_argument('--alpha', type=int, default=1)


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
parser.add_argument('--is_training', type=int, default=1,help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default=modelname,
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')
parser.add_argument('--model_type', type=str, default=modeltype, help='task id')

# forecasting task
parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=pre_len, help='prediction sequence length')
parser.add_argument('--embed_type', type=int, default=0, help='prediction sequence length')
# parser.add_argument('--cross_activation', type=str, default='tanh'

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
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


args = parser.parse_args()
new_d_model = 64
sum_seq_len = {96: 48, 36: 48}

data_parser = {
                'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'alpha': [2], 'hfs':[17, 13], 'k_of_pyramid':[2], 'p_of_pyramid':[2], 'norm':[1], 'ln':[1], 'snet_act':[0], 'ar':[1], 'fft':[True], 'freq':['h']},
                'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'alpha': [2], 'hfs':[17, 13], 'k_of_pyramid':[2], 'p_of_pyramid':[2], 'norm':[1], 'ln':[1], 'snet_act':[0], 'ar':[1], 'fft':[True], 'freq':['h']},
                'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'alpha': [2], 'hfs':[13, 13], 'k_of_pyramid':[4], 'p_of_pyramid':[2], 'norm':[1], 'ln':[1], 'snet_act':[0], 'ar':[1], 'fft':[True], 'freq':['t']},
                'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], 'alpha': [2], 'hfs':[29, 29], 'k_of_pyramid':[4], 'p_of_pyramid':[2], 'norm':[1], 'ln':[1], 'snet_act':[0], 'ar':[1], 'fft':[True], 'freq':['t']},
                'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1], 'idl': [0], 'hfs':[8, 4, 4, 4], 'k_of_pyramid':[3, 3], 'norm':[0], 'ln':[1], 'snet_act':[0], 'ar':[1], 'fft':[True]},
                'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1], 'alpha': [8], 'hfs':[13, 13], 'k_of_pyramid':[4], 'p_of_pyramid':[6], 'norm':[1], 'ln':[1], 'snet_act':[1], 'ar':[1], 'fft':[True], 'freq':['h']},
                'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1], 'idl': [0], 'ln':[1]},
                'ER': {'data': 'ER.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1], 'alpha': [8], 'hfs':[13, 13], 'k_of_pyramid':[4], 'p_of_pyramid':[3], 'norm':[1], 'ln':[0], 'snet_act': [1], 'ar':[1], 'fft':[False], 'freq':['d']},
                'Weather': {'data': 'Weather.csv', 'T': 'CO2 (ppm)', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1], 'alpha': [1], 'hfs': [11, 12, 12], 'k_of_pyramid':[4], 'p_of_pyramid':[3], 'norm': [1], 'ln': [1], 'snet_act': [1], 'ar':[1], 'fft':[True], 'freq':['t']},
                'ILI': {'data': 'ILI.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                'alpha': [1], 'hfs': [49, 37, 1], 'k_of_pyramid':[4], 'p_of_pyramid':[18], 'norm': [1], 'ln': [0], 'snet_act': [1], 'ar':[1], 'fft':[False], 'freq':['7d']},
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
    all_name_of_LRSNet = ['LRSNet', 'LRSNet1', 'LRSNet2']
    Exp = Exp_Main
    # all_task = {'seq_len': [96], 'pre_len': [96, 192, 336, 720], 'dataset_name': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'Weather', 'ER', 'ILI'],
    #             'freq': ['h', 'h', 't', 't','h', 't', 'd', '7d']}
    # ILI, 输入36， 预测[24, 36, 48, 60]
    all_task = {'seq_len': [36], 'pre_len': [96],
                'dataset_name': ['ETTh1']}
    exp_num = 1
    best_all_train_mse = []
    best_all_test_mse = []
    best_all_test_mae = []
    best_all_test_macs = []
    best_all_test_params = []
    best_all_test_avg_time = []
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
                args.alpha = data_info['alpha'][0]
                args.Normalization = data_info['norm'][0]
                args.ar = data_info['ar'][0]
                args.LN = data_info['ln'][0]
                args.snet_act = data_info['snet_act'][0]
                args.fft = data_info['fft'][0]

                args.subseq_len = sum_seq_len[args.seq_len]
                print("subseq_len: ", args.subseq_len)
                args.k_of_pyramid = data_info['k_of_pyramid']
                args.p_of_pyramid = data_info['p_of_pyramid']
                args.freq = data_info['freq'][0]
                args.d_model = new_d_model

                if args.data == 'ECL':
                    pl = np.ones(321)
                else:
                    pl = np.ones(data_info['M'][0])

            df_raw = pd.read_csv(args.root_path + args.data_path)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            y = np.array(df_data)

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

                    setting = '{}_{}_{}_sl{}_pl{}_norm{}_LN{}_act{}_ar{}_fft{}_alpha{}_seed{}_{}'.format(
                        args.model,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.pred_len,
                        args.Normalization,
                        args.LN,
                        args.snet_act,
                        args.ar,
                        args.fft,
                        args.alpha,
                        fix_seed[ii],
                        ii)
                    str_k = ''
                    str_p = ''
                    str_sub_len = str(args.subseq_len)

                    for nn in range(len(args.k_of_pyramid)):
                        str_k += str(args.k_of_pyramid[nn])
                        str_p += str(args.p_of_pyramid[nn])
                    setting += '_' + str_sub_len
                    setting += '_' + str_k
                    setting += '_' + str_p
                    print(args.data, args.model, args.seq_len, args.pred_len, args.batch_size)

                    exp = Exp(args)  # set experiments
                    if args.checkpoint_train:
                        print('>>>>>>>restart checkpoint training : {}'.format(setting))
                        exp.check_train(setting)
                    else:
                        print('>>>>>>>start training : {}'.format(setting))

                        _, train_loss = exp.train(setting)
                        torch.cuda.empty_cache()

                    print('>>>>>>>testing : {}'.format(setting))

                    mse, mae, macs, params, avg_time = exp.test(setting)
                    val_mse, val_mae, _, _, _ = exp.test(setting, 1)

                    if val_mse < best_val_mse:
                        idx = ii
                        best_test_mae = mae
                        best_test_mse = mse
                        best_val_mse = val_mse
                        best_test_macs = macs
                        best_test_params = params
                        best_test_avg_time = avg_time
                        # print(">>"*40, 'mse:', '%.3f' % mse, '  mae:', '%.3f' % mae, ">>"*40, idx)
                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)

                torch.cuda.empty_cache()
                best_all_test_mse.append('%.3f' % best_test_mse)
                best_all_test_mae.append('%.3f' % best_test_mae)
                best_all_test_macs.append('%.4f' % best_test_macs)
                best_all_test_params.append('%.4f' % best_test_params)
                best_all_test_avg_time.append('%.4f' % best_test_avg_time)
                best_all_train_mse.append('%.3f' % train_loss)

                best_all_test_idx.append(idx)
                print(">>" * 40, 'mse:', '%.3f' % best_test_mse, '  mae:', '%.3f' % best_test_mae, ">>" * 10, idx)

            else:
                ii = 0
                random.seed(fix_seed[ii])
                torch.manual_seed(fix_seed[ii])
                np.random.seed(fix_seed[ii])
                setting = '{}_{}_{}_sl{}_pl{}_norm{}_LN{}_act{}_ar{}_fft{}_alpha{}_seed{}_{}'.format(
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.Normalization,
                    args.LN,
                    args.snet_act,
                    args.ar,
                    args.fft,
                    args.alpha,
                    fix_seed[ii],
                    ii)
                str_k = ''
                str_p = ''
                str_sub_len = str(args.subseq_len)

                for nn in range(len(args.k_of_pyramid)):
                    str_k += str(args.k_of_pyramid[nn])
                    str_p += str(args.p_of_pyramid[nn])
                setting += '_' + str_sub_len
                setting += '_' + str_k
                setting += '_' + str_p
                print(args.data, args.model, args.seq_len, args.pred_len, args.batch_size)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting, test=0)   # 0是输入测试集数据，1是输入验证集数据
                exp.predict(setting,load=1)   # 0是输入测试集数据，1是输入验证集数据

                torch.cuda.empty_cache()

    file = r"/home/gzj/former模型/test_results/multi_test_results/multi_test_results.xls"
    oldwb = xlrd.open_workbook(file)  # 打开工作簿
    sheet_names = oldwb.sheet_names()
    newwb = copy(oldwb)
    if modelname not in sheet_names:
        model_sheet = newwb.add_sheet(modelname)
        for i in range(len(best_all_test_mae)):
            row = i + i // len(all_task['pre_len'])  # 核心：每4行数据后增加1个空行偏移
            model_sheet.write(row, 1, best_all_test_mse[i])
            model_sheet.write(row, 2, best_all_test_mae[i])
            model_sheet.write(row, 3, best_all_test_macs[i])
            model_sheet.write(row, 4, best_all_test_params[i])
            model_sheet.write(row, 5, best_all_test_avg_time[i])
            model_sheet.write(row, 6, best_all_train_mse[i])

            newwb.save(file)  # 保存修改
    else:
        model_sheet = newwb.get_sheet(modelname)
        for i in range(len(best_all_test_mae)):
            row = i + i // len(all_task['pre_len'])  # 核心：每4行数据后增加1个空行偏移
            model_sheet.write(row, 1, best_all_test_mse[i])
            model_sheet.write(row, 2, best_all_test_mae[i])
            model_sheet.write(row, 3, best_all_test_macs[i])
            model_sheet.write(row, 4, best_all_test_params[i])
            model_sheet.write(row, 5, best_all_test_avg_time[i])
            model_sheet.write(row, 6, best_all_train_mse[i])

            newwb.save(file)  # 保存修改


