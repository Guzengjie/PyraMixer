# PyraMixer

PyraMixer is a multi-scale time series forecasting model based on a pyramid architecture. This codebase implements a complete time series forecasting framework supporting various advanced deep learning models for multivariate time series prediction tasks.

## Project Overview

PyraMixer employs an innovative pyramid-based multi-scale hybrid architecture to achieve high-accuracy long-term forecasting. The core idea of the model is to decompose the input sequence into multiple scale levels using interval sampling, then extract features and perform forecasting with MSM.

## Model Architecture

### PyraMixer Core Components

- **Splitting**: Sequence decomposition module that divides the input time series into even and odd parts
- **DGS**: Deep Gating Module for feature selection and filtering
- **FFTMix**: Frequency-domain feature mixer based on Fast Fourier Transform
- **SampleLinear**: Sampling linear transformation layer
- **MSM**: Multi-Scale Mixing module
- **LRSNU**: Long Sequence Update Network

### Supported Models

| Model Name | Description |
|------------|-------------|
| PyraMixer | Pyramid multi-scale hybrid model |
| DLinear | Linear model with seasonal decomposition |
| FEDformer | Transformer-based on Fourier transform |
| LightTS | Lightweight time series model |
| PatchTST | Patch-based time series Transformer |
| SCINet | Sparse Convolutional Interaction Network |
| TimeMixer | Time mixing model |
| TimesNet | Time-frequency transformation network |
| AMD | Adaptive Mixed Deep Network |

## Datasets

The project natively supports several classic time series forecasting datasets:

- **ETTh1/ETTh2**: Hourly electricity transformer data
- **ETTm1/ETTm2**: Minute-level electricity transformer data
- **ECL**: Electricity consumption dataset
- **Weather**: Meteorological data
- **ILI**: Influenza-like illness data
- **ER**: Energy-related data

## Environment Requirements

```
torch~=2.7.0+cu128
numpy~=2.1.2
matplotlib~=3.10.5
sympy~=1.13.3
torchinfo~=1.8.0
thop~=0.1.1-2209072238
ultralytics-thop~=2.0.17
scipy~=1.15.3
pandas~=2.3.1
ultralytics~=8.3.203
scikit-learn~=1.7.1
xlrd~=2.0.2
xlutils~=2.0.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train PyraMixer Model

```bash
cd run_LTSF
python run_PyraMixer.py
```

### Train Other Models

```bash
# Train DLinear
python run_DLinear.py

# Train PatchTST
python run_PatchTST.py

# Train TimesNet
python run_TimeMixer.py
```

### Scripts

We provide easy-to-run shell scripts (.sh files) in the scripts/ folder for training and evaluation on different datasets. You can directly run the scripts to reproduce experiments without manually setting parameters.

```bash
bash scripts/PyraMixer_Long.sh

## Parameter Configuration

Key training parameters include:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| --is_training | Training mode | 1 |
| --model_id | Model identifier | test |
| --model | Model name | PyraMixer |
| --data | Dataset name | ETTh1 |
| --features | Prediction type | M |
| --seq_len | Input sequence length | 96 |
| --label_len | Label length | 48 |
| --pred_len | Prediction length | 96 |
| --enc_in | Encoder input dimension | 7 |
| --dec_in | Decoder input dimension | 7 |
| --c_out | Output dimension | 7 |
| --d_model | Model dimension | 512 |
| --n_heads | Number of attention heads | 8 |
| --e_layers | Number of encoder layers | 2 |
| --d_layers | Number of decoder layers | 1 |
| --d_ff | Feedforward network dimension | 2048 |
| --moving_avg | Moving average window | 25 |
| --factor | Attention factor | 1 |
| --distil | Distillation factor | True |
| --dropout | Dropout rate | 0.1 |
| --embed | Embedding type | timeF |
| --activation | Activation function | gelu |
| --output_attention | Output attention | False |
| --do_predict | Prediction mode | False |

## Project Structure

```
.
├── data_provider/          # Data loading module
│   ├── data_factory.py
│   └── data_loader_utf8.py
├── datasets/               # Dataset files
├── exp/                   # Experiment base class
│   ├── exp_basic.py
│   └── exp_main_LTSF.py
├── layers/                # Neural network layers
│   ├── AutoCorrelation.py
│   ├── Autoformer_EncDec.py
│   ├── Embed.py
│   ├── FourierCorrelation.py
│   ├── MultiWaveletCorrelation.py
│   ├── PatchTST_backbone.py
│   ├── RevIN.py
│   ├── SelfAttention_Family.py
│   └── Transformer_EncDec.py
├── models/                # Forecasting models
│   ├── PyraMixer.py
│   ├── DLinear.py
│   ├── FEDformer.py
│   ├── LightTS.py
│   ├── PatchTST.py
│   ├── SCINet.py
│   ├── TimeMixer.py
│   ├── TimesNet.py
│   └── AMD.py
├── run_LTSF/              # Training scripts
├── utils/                 # Utility functions
│   ├── metrics.py
│   ├── tools.py
│   ├── masking.py
│   └── timefeatures.py
└── requirements.txt
```

## Evaluation Metrics

The project uses the following metrics for model evaluation:

- **RSE**: Root Square Error
- **CORR**: Correlation coefficient
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **MSPE**: Mean Squared Percentage Error

## Citation

If this project aids your research, please cite the relevant paper:

PyraMixer is built upon the latest research in multi-scale time series analysis.

## License

This project is intended solely for academic research purposes.
