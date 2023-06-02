#! /bin/sh
touch logs/log_ot_cls.txt

nohup python main.py \
    --name ot_cls \
    --dataset factor_celeba \
    --train True \
    --test False \
    --model ot_cls \
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
    --eta1 0.1 \
    --eta2 0.4 \
    --n_sens 1 \
    --sens_idx 20 \
    --ot_idx 2 \
    --ckpt_load last \
    --sens_cls_name sens_cls \
    --sens_ckpt 34 \
    > logs/log_ot_cls.txt &
    $@
