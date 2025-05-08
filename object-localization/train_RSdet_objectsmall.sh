CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=12342 \
/data/XJL/open-vocabulary/DINO-main/main.py \
--pretrain_model_path \
/data/XJL/open-vocabulary/DINO-main/ckpt/checkpoint0033_4scale.pth \
--output_dir \
/data/XJL/dino/DINO-main/out/output36_RSdet_objectsmall \
-c \
/data/XJL/open-vocabulary/DINO-main/config/DINO/DINO_4scale.py \
--coco_path \
/data/XJL/dataset/RSdet_datasetsmall \
--options \
dn_scalar=100 \
embed_init_tgt=TRUE \
dn_label_coef=1.0 \
dn_bbox_coef=1.0 \
use_ema=False \
dn_box_noise_scale=1.0