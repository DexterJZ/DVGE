#! /bin/sh
touch logs/log_sens_cls1.txt

nohup python main.py \
    --name sens_cls1 \
    --dataset factor_celeba \
    --train True \
    --test False \
    --model sens_cls \
    --gpu 0 \
    --num_workers 4 \
    --batch_size 64 \
    --output_save True \
    --ckpt_save_iter 10000 \
    --max_iter 1e6 \
    --lr_D 1e-5 \
    --beta1_D 0.5 \
    --beta2_D 0.9 \
    --z_dim 10 \
    --n_sens 2 \
    --sens_idx 20 39 \
    --ckpt_load last \
    > logs/log_sens_cls1.txt &
    $@
