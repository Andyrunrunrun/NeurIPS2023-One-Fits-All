
export CUDA_VISIBLE_DEVICES=5

seq_len=336
model=MultiModelTS

for percent in 100
do
for pred_len in 96
do
for lr in 0.0001
do

python main.py \
    --root_path /chenyaofo/andy/new_project/NeurIPS2023-One-Fits-All/datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$model_layer'_'$seq_len'_'$pred_len'_'$percent \
    --base_model gpt2 \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 128 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --model_layer 100 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --base_model /chenyaofo/hf_models/Llama-2-7b-hf

done
done
done
