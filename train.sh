python main.py \
    -t \
    --base configs/train.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --logdir "logs" \
    --finetune_from pre_trained/sd-image-variations-diffusers/sd-image-conditioned-v2.ckpt \
