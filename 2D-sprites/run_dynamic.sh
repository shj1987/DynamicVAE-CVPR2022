#! /bin/sh
  
python3 main.py --train True --dataset dsprites --seed 6 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 128 --z_dim 10 --max_iter 2e6 \
    --C_stop_iter 2e6 --step_val 0.15 --gpu 0 \
    --viz_name dsprites_Dynamic_IPI_c20_period5_v7 --C_max 20 --Ki -0.005 --past_T 5;
    