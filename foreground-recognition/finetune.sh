#!/usr/bin/env bash

OUTPUT_DIR='/data/XJL/LLM/DINO-LLAMA930/LLAMA-Adapter-v3/output/oddataset'
mkdir -p "$OUTPUT_DIR"

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
CUDA_VISIBLE_DEVICES=2,3  python -u -m torch.distributed.launch --master_port=1113 --nproc_per_node=2 --use_env main_finetune.py --batch_size 4 \
   --epochs 10 --warmup_epochs 1 --blr 1e-5 --weight_decay 0.02 \
   --output_dir "$OUTPUT_DIR" \
   --pretrained_path /data/XJL/LLM/DINO-LLAMA930/LLAMA-Adapter-v3/output/oddataset/checkpoint-1-7B.pth \
   --llama_path /data/XJL/LLM/7B-chinese-params \
   --start_epoch 2 \
   --max_words 512
