#! /bin/sh
touch logs/log_factor_celebat.txt

nohup python main.py \
    --name factor_celeba \
    --dataset celeba \
    --train False \
    --test True \
    --model factor \
    --gpu 0 \
    --num_workers 4 \
    --batch_size 1 \
    --output_save True \
    --ckpt_save_iter 10000 \
    --max_iter 1e6 \
    --lr_VAE 1e-4 \
    --beta1_VAE 0.9 \
    --beta2_VAE 0.999 \
    --lr_D 1e-5 \
    --beta1_D 0.5 \
    --beta2_D 0.9 \
    --z_dim 10 \
    --gamma 6.4 \
    --ckpt_load last \
    > logs/log_factor_celebat.txt &
    $@
