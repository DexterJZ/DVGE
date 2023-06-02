#! /bin/sh
touch logs/log_beta_celebat.txt

nohup python main.py \
    --name beta_celeba \
    --dataset celeba \
    --train False \
    --test True \
    --model beta \
    --gpu 0 \
    --num_workers 4 \
    --batch_size 1 \
    --output_save True \
    --ckpt_save_iter 10000 \
    --max_iter 1e6 \
    --lr_VAE 1e-4 \
    --beta1_VAE 0.9 \
    --beta2_VAE 0.999 \
    --z_dim 10 \
    --beta 4.0 \
    --ckpt_load last \
    > logs/log_beta_celebat.txt &
    $@
