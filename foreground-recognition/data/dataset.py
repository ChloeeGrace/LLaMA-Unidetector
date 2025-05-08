import torch
import torchvision
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def transform_train_clip_func(tensor):
    # Assuming tensor is (C, H, W) and in the range [0, 1]
    tensor = F.resize(tensor, (336, 336))
    tensor = F.normalize(tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return tensor

def transform_train_dino_func(tensor):
    # Assuming tensor is (C, H, W) and in the range [0, 1]
    tensor = F.resize(tensor, (336, 336))
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor



# create data
transform_train_clip = transforms.Compose([
    # transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    transforms.Resize((336,336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


transform_train_dino = transforms.Compose([
    # transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    transforms.Resize((336,336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer_path=None):

        pathprefix = "/data/XJL/dataset/data_json"
        
    
        
        # =============== llava finetune  ========================
        ann1 = json.load(open(os.path.join(pathprefix, 'DIOR_LLAVA_TUNE.json')))
        ann3 = json.load(open(os.path.join(pathprefix, 'DOTA_LLAVA_TUNE.json')))
        ann4 = json.load(open(os.path.join(pathprefix, 'NWPU45_LLAVA_TUNE.json')))
        ann5 = json.load(open(os.path.join(pathprefix, 'AID30_LLAVA_TUNE.json')))
        # ========================================================

        ann2 = json.load(open(os.path.join(pathprefix, 'alpaca_gpt4_data.json')))
        ann6 = json.load(open(os.path.join(pathprefix, 'alpaca_gpt4_data_zh.json')))
        ann7 = json.load(open(os.path.join(pathprefix, 'unnatural_instruction_gpt4_data.json')))
        ann8 = json.load(open(os.path.join(pathprefix, 'llava_instruct_150k_split_bits.json')))

        # ann9 = json.load(open(os.path.join(pathprefix, 'comparison_data_v2.json')))

        self.ann = ann1 + ann2 + ann3 + ann6 + ann7 + ann8
        self.lang_type = ['EN'] * len(ann1) + ['EN'] * len(ann2) + ['EN'] * len(ann3) + ['CH'] * len(ann6) + ['EN'] * len(ann7) + ['EN'] * len(ann8)

        # self.ann = ann1 + ann2  + ann3 + ann4 + ann5 + ann6 + ann7 + ann8
        # self.lang_type = ['EN'] * len(ann1) + ['EN'] * len(ann2) + ['EN'] * len(ann3) + ['EN'] * len(ann4) + ['EN'] * len(ann5) + ['CH'] * len(ann6) + ['EN'] * len(ann7) + ['EN'] * len(ann8)
        # self.ann = ann2 + ann4 + ann6 + ann7
        # self.lang_type = ['EN'] * len(ann2) + ['EN'] * len(ann4) + ['CH'] * len(ann6) + ['EH'] * len(ann7)



        print(f"total length: {len(self)}")
        self.transform_clip = transform_train_clip
        # self.transform_dino = transform_train_dino
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]

        # ======== original ======
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = str(data_item['conversations'][1]['value'])


            # < fill path substitution logics here>
            # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")

            image = Image.open(filename).convert('RGB')
            image_clip = self.transform_clip(image)
            # image_dino = self.transform_dino(image)
            format_instruction = question
            format_input = None
        else:
            image_clip = torch.zeros(3, 336, 336)
            # image_dino = torch.zeros(3, 336, 336)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        # =========================


        # ====== for coco caption finetune ======
        # if 'img-tensor' in data_item.keys(): #是coco-caption数据集中的数据
        #     # filename = data_item['image'].replace('/data0/data', '/mnt/petrelfs/share_data/hanjiaming')
        #     #filename = '/data/coco-2014/train2014/' + data_item['image']
        #     question = "Generate caption for this image."
        #     #answer = "\n".join(data_item['cap-list'])
        #     answer = str(data_item['cap-list'][0])
        #     # < fill path substitution logics here>
        #     # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")
        #     image = data_item['img-tensor'].float()
        #     image_clip = transform_train_clip_func(image)
        #     image_dino = transform_train_dino_func(image)
            
        #     format_instruction = question
        #     format_input = None
        # else:
        #     image_clip = torch.zeros(3, 336, 336)
        #     image_dino = torch.zeros(3, 336, 336)
        #     format_instruction = data_item['instruction'],
        #     format_input = data_item['input']
        #     answer = data_item['output']
        # ======================================
        
        
        
        input1 = llama.utils.format_prompt(format_instruction, format_input, self.lang_type[index])
        input2 = input1 + answer # wen da
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        
        
        
        
        
        
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image_clip


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        
        
        # =========  original  ===========================
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        
        
        
        
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
