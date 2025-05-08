# encoding: utf-8

import cv2
import llama
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import torchvision.transforms as T
from tqdm import tqdm
import json
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="test performance params")
    parser.add_argument("--index", default=1, type=int, help="sub input image directory path")
    return parser

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.Resize((336,336)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_clip = T.Compose(
       [
    # transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    T.Resize((336,336)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
    )


    # image = transform(image_pil)
    image_clip = transform_clip(image_pil)
    return image_clip



def test_performance(val_dir, model,prompt,preprocess,device):
    
    imgs = os.listdir(val_dir)
    results = []
    for img_name in tqdm(imgs):
        res_dict = {}
        image_path = os.path.join(val_dir,img_name)
        #---------------------------------
        img_clip = load_image(image_path)
        # img_dino = img_dino.unsqueeze(0).to(device)
        # img_clip = torch.zeros(3, 336, 336)
        img_clip = img_clip.unsqueeze(0).to(device)
        #——------------------------------
        
        result,prob = model.generate(img_clip, [prompt])
        image_id = int(img_name.split("_")[-1].split(".")[0])
        res_dict["image_id"] = image_id
        res_dict["detection"] = result[0]
        res_dict["probability"] = prob.item()
        results.append(res_dict)
        print(img_name, results)
    results.sort(key=lambda x: x['image_id'])
    return results


def test_performance_text(category, model, prompt, preprocess, device):
    img_clip = torch.zeros(3, 336, 336)
    img_clip = img_clip.unsqueeze(0).to(device)
    # ——------------------------------
    result = model.generate(img_clip, [prompt])[0]
    print(category, result)
    # results.sort(key=lambda x: x['image_id'])
    return result

if __name__ ==  "__main__":

    args = get_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llama_dir = "/data/XJL/LLM/7B-chinese-params"
    model, preprocess = llama.load("/data/XJL/github/LLaMA-Unidetector/foreground-recognition/TerraOV-LLM/output/8dataset/checkpoint-3-7B.pth", llama_dir, device)
    model.eval()



    # multi-modal
    text_prompt = 'What type of remote sensing scene or object is depicted in this image?'
    # text_prompt = 'What name of remote sensing scene or object can be described in this image by a word in the list below?'
    # text_prompt = 'Based on the visual features in the image, which category or object do you think this scene belongs to'

    prompt = llama.format_prompt(text_prompt)
    # # val_dir = "/data/Datasets/debug_llm"
    # result folder

    result_txt_folder = '/data/XJL/open-vocabulary/object-localization/out/test/llama_txt/out_RSdet_objectsmall_dota_txt/fb_od_0.22'
    if not os.path.exists(result_txt_folder):
        os.mkdir(result_txt_folder)

    # det results txt folder
    det_txt_folder = '/data/XJL/open-vocabulary/object-localization/out/test/txt/RSdet_objectsmall_dota_txt'

    # one img foler for per detection img
    img_folders = '/data/XJL/open-vocabulary/obejct-loaclization/out/test/slices/slices_RSdet_objectsmall_dota/fb_enlarging_0.22'
    img_folders_list = os.listdir(img_folders)
    for img_folder_name in tqdm(img_folders_list):
        # val_dir = ''
        img_folder_path = os.path.join(img_folders, img_folder_name)

        results = test_performance(img_folder_path, model, prompt, preprocess, device)
        txt_path_name = img_folder_name + '.txt'
        result_file_txt = os.path.join(result_txt_folder, txt_path_name)
        score_coord_txt = os.path.join(det_txt_folder, txt_path_name)

        with open(score_coord_txt, "r") as f1:
            lines = f1.readlines()

        # RSdet_objectsmall
        f0 = open(result_file_txt, "w")
        for i, result in enumerate(results):
            line = lines[i].strip().split(' ')
            score = line[1]
            x1, y1, x2, y2 = line[2], line[3], line[4], line[5]
            score2 = result['probability']
            f0.write(
                "%s %s %s %s %s %s %s\n" % (result['detection'], float(score), float(score2), float(x1), float(y1), float(x2), float(y2)))
        f0.close()
