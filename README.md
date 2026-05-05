Official implementation of PyraMixer, a pyramid-based mixer architecture for efficient Long-Term Time Series Forecasting (LTSF).
Table of Contents
Introduction
Requirements
Project Structure
Quick Start
Results
Citation

1. Introduction
This repository contains the implementation of PyraMixer for Long-Term Time Series Forecasting (LTSF).It adopts pyramid multi-scale feature extraction and lightweight mixer design to capture long-range temporal dependencies with low computational cost.

2. Requirements
plaintext
torch>=1.10.0
numpy
pandas
scikit-learn
matplotlib
tqdm

Install:
bash
运行
pip install -r requirements.txt

3. Project Structure
plaintext
PyraMixer/
├── data_provider/     # Data loading & preprocessing
├── datasets/          # Raw time series datasets
├── layers/            # Pyramid & mixer basic blocks
├── models/            # PyraMixer model definition
├── exp/               # Training & evaluation pipeline
├── utils/             # Metrics and tool functions
├── scripts/           # Running scripts
├── run_LTSF/          # Main entry
└── requirements.txt

4. Quick Start
Train
bash运行
bash scripts/etth1.sh

bash运行
python run_LTSF/run.py
