import numpy as np
import matplotlib.pyplot as plt

data_dict = {
    'ETTh2_true': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTh2_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_2_2/real_trues.npy',
    'PyraMixer_ETTh2_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTh2_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_2_2/real_prediction.npy',
    'AMD_ETTh2_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ETTh2_M_sl96_pl192_b32_p4_k3_c2_alpha0.0_normTrue_seedFalse_itr2024/real_prediction.npy',
    'TimeMixer_ETTh2_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ETTh2_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ETTh2_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ETTh2_M_sl96_ll96_pl192_dm128_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'Electricity_true': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ECL_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha8_seed2024_0_48_4_6/real_trues.npy',
    'PyraMixer_Electricity_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ECL_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha8_seed2024_0_48_4_6/real_prediction.npy',
    'AMD_Electricity_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ECL_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedFalse_itr2024/real_prediction.npy',
    'TimeMixer_Electricity_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ECL_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_Electricity_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ECL_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',


    'Weather_true': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_Weather_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha1_seed2024_0_48_4_3/real_trues.npy',
    'PyraMixer_Weather_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_Weather_M_sl96_pl192_norm1_LN1_act1_ar1_fftTrue_alpha1_seed2024_0_48_4_3/real_prediction.npy',
    'AMD_Weather_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_Weather_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_Weather_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_Weather_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_Weather_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_Weather_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ILI_true': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ILI_M_sl36_pl60_norm1_LN0_act1_ar1_fftFalse_alpha1_seed2024_0_48_4_18/real_trues.npy',
    'PyraMixer_ILI_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ILI_M_sl36_pl60_norm1_LN0_act1_ar1_fftFalse_alpha1_seed2024_0_48_4_18/real_prediction.npy',
    'AMD_ILI_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ILI_M_sl36_pl60_b32_p4_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_ILI_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ILI_M_sl36_ll36_pl60_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ILI_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ILI_M_sl36_ll36_pl60_dm128_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'Exchange_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ER_M_sl96_pl192_norm1_LN0_act1_ar1_fftFalse_alpha8_seed2024_0_48_4_3/real_trues.npy',
    'PyraMixer_Exchange_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ER_M_sl96_pl192_norm1_LN0_act1_ar1_fftFalse_alpha8_seed2024_0_48_4_3/real_prediction.npy',
    'AMD_Exchange_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ER_M_sl96_pl192_b32_p4_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_Exchange_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ER_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_Exchange_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ER_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy',

    'ETTm1_true':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTm1_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_4_2/real_trues.npy',
    'PyraMixer_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/PyraMixer/PyraMixer_ETTm1_M_sl96_pl192_norm1_LN1_act0_ar1_fftTrue_alpha2_seed2024_0_48_4_2/real_prediction.npy',
    'AMD_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/AMD/AMD_ETTm1_M_sl96_pl192_b32_p16_k3_c2_alpha0.0_normTrue_seedTrue_itr2024/real_prediction.npy',
    'TimeMixer_ETTm1_pred':'/home/gzj/former模型/run_LTSF/results/real_prediction/TimeMixer/TimeMixer_ETTm1_M_sl96_ll96_pl192_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_seed2024_itr0/real_prediction.npy',
    'PatchTST_ETTm1_pred': '/home/gzj/former模型/run_LTSF/results/real_prediction/PatchTST/PatchTST_ETTm1_M_sl96_ll96_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_1_seed2024_0/real_prediction.npy'

}

# 颜色设置
colors = {
    'true': '#000000',      # 黑色 - 真实值
    'PyraMixer': '#FF6B35',  # 红色
    'AMD': '#3498DB',        # 蓝色
    'TimeMixer': '#9B59B6',  # 绿色
    'PatchTST': '#2ECC71'    # 紫色
}

# 模型名称映射（用于图例）
model_names = {
    'PyraMixer': 'PyraMixer',
    'AMD': 'AMD',
    'TimeMixer': 'TimeMixer',
    'PatchTST': 'PatchTST'
}

sample_feature ={
    'ETTh2':[1241, 2],
    'Electricity':[1024,36],
    'Weather':[1024,20],
    'ILI':[66, 2],
    'ETTm1':[1241, 2],
    'Exchange':[128,3]
}

def load_data(key):
    # 替换为你的实际加载代码，例如：
    return np.load(data_dict[key])

# 加载所有数据
data = {}
for key, path in data_dict.items():
    data[key] = load_data(key)

# 字体设置（Times New Roman是SCI标准）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也统一
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7

# 创建1×3子图
fig, axes = plt.subplots(2, 3, figsize=(7.48, 5), dpi=300)

datasets = ['Electricity', 'Weather', 'ETTh2', 'Exchange', 'ETTm1','ILI']
models = ['PyraMixer', 'AMD', 'TimeMixer', 'PatchTST']

# 存储线条句柄（用于图例）
handles_dict = []
for idx, dataset in enumerate(datasets):
    # ax = axes[idx]
    ax = axes.flat[idx]

    # 绘制真实值
    true_key = f'{dataset}_true'

    # 绘图
    ground_truths_sample = data[true_key][sample_feature[dataset][0], :, sample_feature[dataset][1]]
    time_steps = np.arange(len(ground_truths_sample))
    line, = ax.plot(time_steps, ground_truths_sample, color=colors['true'], linewidth=0.8,
            label='GroundTruth', alpha=0.9)
    if idx==0:
        handles_dict.append(line)

    # 绘制各模型预测
    for model in models:
        pred_key = f'{model}_{dataset}_pred'
        pred_sample = data[pred_key][sample_feature[dataset][0], :, sample_feature[dataset][1]]

        line, = ax.plot(time_steps,pred_sample, color=colors[model], linewidth=0.8,
                label=model_names[model], alpha=0.8)
            # 只保存第一次出现的句柄
        if model not in handles_dict:
            handles_dict.append(line)

    # 设置子图标题和标签
    ax.set_xlabel(f'{dataset}')

# 关键：在整张图上方添加统一图例（1行6列）
labels = ['GroundTruth', 'PyraMixer', 'AMD', 'TimeMixer', 'PatchTST']

fig.legend(handles_dict, labels, loc='upper center',
           ncol=6,  # 1行6列
           frameon=False,
           bbox_to_anchor=(0.5, 1),  # 位置：图正上方居中
           columnspacing=1.5, handletextpad=0.5)

# 调整布局，给顶部图例留空间
plt.tight_layout(rect=[0.0, 0.0, 1, 0.985])  # top=0.96 给图例留4%空间
plt.subplots_adjust(wspace=0.2, hspace=0.2)  # 子图间距

# 保存（PDF矢量图用于投稿，TIFF用于审稿）
plt.savefig('fig_forecast_comparison.pdf', format='pdf', dpi=600,
           bbox_inches='tight', facecolor='white', edgecolor='none')

plt.savefig('fig_forecast_comparison.tiff', format='tiff', dpi=600,
           bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()



