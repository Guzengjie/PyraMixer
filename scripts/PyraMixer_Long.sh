# ALL scripts in this file come from Autoformer
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in PyraMixer
do
for pred_len in 96 192 336 720
do
  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
    --is_training 1 \
    --seq_len 96 \
    --data 'ER'
    --train_epochs 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
    --is_training 1 \
    --seq_len 96 \
    --data 'ETTh1'
    --train_epochs 1 >logs/LongForecasting/$model_name'_ETTh1_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 96 \
  --data 'ETTh2'
  --train_epochs 1 >logs/LongForecasting/$model_name'_ETTh2_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 96 \
  --data 'ETTm1'
  --train_epochs 1 >logs/LongForecasting/$model_name'_ETTm1_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 96 \
  --data 'ETTm2'
  --train_epochs 1 >logs/LongForecasting/$model_name'_ETTm2_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 96 \
  --data 'ECL'
  --train_epochs 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len.log

  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 96 \
  --data 'Weather'
  --train_epochs 1 >logs/LongForecasting/$model_name'_Weather_'$pred_len.log

done
done

for model_name in PyraMixer
do
for pred_len in 24 36 48 60
do
  python -u /home/gzj/PycharmProjects/PyraMixer/run_LTSF/run.py \
  --is_training 1 \
  --seq_len 36 \
  --data 'ILI'
  --train_epochs 1 >logs/LongForecasting/$model_name'_ILI_'$pred_len.log
done
done
