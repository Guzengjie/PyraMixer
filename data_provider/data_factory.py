from data_provider.data_loader_utf8 import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_ECL_hour, Dataset_WTH_hour, \
    Dataset_ER_day, Dataset_SAL_minute, Dataset_Weather_minute, Dataset_ILI_week, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_ECL_hour,
    'WTH': Dataset_WTH_hour,
    'ER': Dataset_ER_day,
    'solar_AL': Dataset_SAL_minute,
    'Weather': Dataset_Weather_minute,
    'ILI': Dataset_ILI_week
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    # print(flag, timeenc)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    # print(flag, "——批次数量：", len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
