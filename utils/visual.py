import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ultralytics.utils.metrics import bbox_ioa

# 设置中文字体（如需中文标签）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False



def visual_four(m1_pre,m2_pre,m3_pre,m4_pre, true_path, sample_idx, feature_idx, pred_length=192):
    # -------------------------- 2. 可视化多模型预测曲线 --------------------------

    plt.figure(figsize=(16, 8))

    # 加载数据
    m1_pre = np.load(m1_pre)  # [N, pred_len, features]
    m2_pre = np.load(m2_pre)  # [N, pred_len, features]
    m3_pre = np.load(m3_pre)  # [N, pred_len, features]
    m4_pre = np.load(m4_pre)  # [N, pred_len, features]
    ground_truths = np.load(true_path)

    print(f"预测值形状: {m1_pre.shape}")
    print(f"样本数: {m1_pre.shape[0]}, 预测步长: {m1_pre.shape[1]}, 特征数: {m1_pre.shape[2]}")

    m1_pred_sample = m1_pre[sample_idx, :, feature_idx]
    m2_pred_sample = m2_pre[sample_idx, :, feature_idx]
    m3_pred_sample = m3_pre[sample_idx, :, feature_idx]
    m4_pred_sample = m4_pre[sample_idx, :, feature_idx]
    ground_truths_sample = ground_truths[sample_idx, :, feature_idx]

    # 绘图
    time_steps = np.arange(len(ground_truths_sample))

    # 绘制曲线：区分历史段、预测段，不同模型用不同颜色/线型
    # 1. 真实值（历史+预测）
    plt.plot(time_steps, ground_truths_sample, color='black', linewidth=2.5, label='Ground Truth', zorder=5)

    # 2. 4个模型的预测曲线（仅预测段显示）
    plt.plot(time_steps, m1_pred_sample, color='#2E86AB', linewidth=2, linestyle='-', label='PyraMixer', zorder=4)
    plt.plot(time_steps, m2_pred_sample, color='#A23B72', linewidth=2, linestyle='--', label='AMD', zorder=3)
    plt.plot(time_steps, m3_pred_sample, color='#F18F01', linewidth=2, linestyle='-.', label='TimeMxier', zorder=2)
    plt.plot(time_steps, m4_pred_sample, color='#C73E1D', linewidth=2, linestyle=':', label='PatchTST', zorder=1)
    # -------------------------- 3. 图表美化与标注 --------------------------
    # plt.title(f'4个模型长期时间序列预测对比（预测长度={pred_length}）', fontsize=16, pad=20)
    plt.xlabel('Fulture Time Steps', fontsize=12)
    plt.ylabel('Prediction Results', fontsize=12)

    # 优化x轴刻度（避免长序列标签重叠）
    step = max(1, pred_length // 20)  # 控制刻度数量
    plt.xticks(np.arange(0, pred_length, step), rotation=45)

    plt.grid(True, linestyle='--', alpha=0.5)  # 网格线提升可读性
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncols=5, fontsize=10, borderaxespad=0)  # 图例位置
    plt.tight_layout()  # 自适应布局（防止标签截断）
    plt.show()

data_dict = {
    'ETTh2_true': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTh2_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_2_2/real_trues.npy',
    'PyraMixer_ETTh2_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTh2_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_2_2/real_prediction.npy',
    'AMD_ETTh2_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ETTh2_M_sl96_pl192_b32_p4_k3_c2_alpha0.0_normTrue_seedFalse_itr2024/real_prediction.npy',
    'TimeMixer_ETTh2_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ETTh2_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ETTh2_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ETTh2_M_sl96_ll96_pl192_dm128_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ECL_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ECL_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha8_seed2024_0_48_4_6/real_trues.npy',
    'PyraMixer_ECL_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ECL_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha8_seed2024_0_48_4_6/real_prediction.npy',
    'AMD_ECL_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ECL_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedFalse_itr2024/real_prediction.npy',
    'TimeMixer_ECL_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ECL_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ECL_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ECL_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'Weather_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_Weather_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha1_seed2024_0_48_4_3/real_trues.npy',
    'PyraMixer_Weather_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_Weather_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha1_seed2024_0_48_4_3/real_prediction.npy',
    'AMD_Weather_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_Weather_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_Weather_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_Weather_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_Weather_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_Weather_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ILI_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ILI_M_sl36_pl60_norm1_LN0_act1_ar1_fftFalse_alpha1_seed2024_0_48_4_18/real_trues.npy',
    'PyraMixer_ILI_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ILI_M_sl36_pl60_norm1_LN0_act1_ar1_fftFalse_alpha1_seed2024_0_48_4_18/real_prediction.npy',
    'AMD_ILI_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ILI_M_sl36_pl60_b32_p4_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_ILI_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ILI_M_sl36_ll36_pl60_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ILI_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ILI_M_sl36_ll36_pl60_dm128_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ER_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ER_M_sl96_pl192_norm1_LN0_act1_ar1_fftFalse_alpha8_seed2024_0_48_4_3/real_trues.npy',
    'PyraMixer_ER_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ER_M_sl96_pl192_norm1_LN0_act1_ar1_fftFalse_alpha8_seed2024_0_48_4_3/real_prediction.npy',
    'AMD_ER_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ER_M_sl96_pl192_b32_p4_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_ER_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ER_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ER_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ER_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ETTm1_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTm1_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_4_2/real_trues.npy',
    'PyraMixer_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTm1_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_4_2/real_prediction.npy',
    'AMD_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ETTm1_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ETTm1_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ETTm1_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ETTm1_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy'

}

sample_feature ={
    'ETTh2':[1241, 2],
    'ECL':[66,36],
    'Weather':[66,2],
    'ILI':[66, 2],
    'ETTm1':[1241, 2],
    'ER':[128,3]
}

model_names = ['PyraMixer', 'AMD', 'TimeMixer', 'PatchTST']
def model_data(dataset_name, model_names):
    pres_and_true = []
    for i in range(len(model_names)):
        pre = model_names[i]+'_'+dataset_name+'_pred'
        pres_and_true.append(pre)
    true = dataset_name+'_true'
    pres_and_true.append(true)
    return pres_and_true
dataset_name = 'ER'
keys = model_data(dataset_name, model_names)
visual_four(data_dict[keys[0]], data_dict[keys[1]], data_dict[keys[2]], data_dict[keys[3]], data_dict[keys[4]],  sample_feature[dataset_name][0], sample_feature[dataset_name][1])
print(keys)
print(dataset_name, sample_feature[dataset_name])