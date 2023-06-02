#! /bin/sh
touch logs/log_ff_celeba4.txt

nohup python main.py \
    --name ff_celeba4 \
    --dataset celeba \
    --train True \
    --test False \
    --model ff \
    --gpu 0 \
    --num_workers 4 \
    --batch_size 64 \
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
    --gamma 1.0 \
    --beta 4.0 \
    --alpha 1.0 \
    --n_sens 2 \
    --sens_idx 20 39 \
    --ckpt_load last \
    > logs/log_ff_celeba4.txt &
    $@
