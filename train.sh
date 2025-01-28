cd /home/lixin/projects/zero123_ccgen/zero123
python main.py \
    -t \
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --logdir "/memory/a100/outputs/zero123/logs_ccgen_ablation" \
    --finetune_from /memory/a100/pre_trained_sdxl/sd-image-variations-diffusers/sd-image-conditioned-v2.ckpt \
    # --resume='/memory/a100/outputs/zero123/logs_ccgen_inpainting/2025-01-13T18-31-46_sd-objaverse-finetune-c_concat-256/checkpoints/checkpoint40000.ckpt' \