CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port \
12345 \
/data/XJL/open-vocabulary/object-localization/main.py \
--output_dir \
/data/XJL/open-vocabulary/object-localization/out/test/log/log_RSdet_small_dior \
-c \
/data/XJL/open-vocabulary/object-localization/config/DINO/DINO_4scale.py \
--coco_path \
/data/XJL/dataset/RSdet_datasetsmall_finetune \
--eval \
--resume \
/data/XJL/dino/DINO-main/out/output36_fb_dior/checkpoint_best_regular.pth \
--options \
dn_scalar=100 \
embed_init_tgt=TRUE \
dn_label_coef=1.0 \
dn_bbox_coef=1.0 \
use_ema=False \
dn_box_noise_scale=1.0