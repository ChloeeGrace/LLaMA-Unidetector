import torch
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

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(336, 336), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, max_words=30, tokenizer_path=None):

        # ann1 = json.load(open(os.path.join('/data/Datasets/Vision-Language','llava_instruct_150k_single_turn.json')))
        ann1 = json.load(open(os.path.join('/data/Datasets/Vision-Language', 'llava_instruct_150k.json')))
        ann2 = json.load(open(os.path.join('/data/Datasets/Vision-Language','alpaca_gpt4_data.json')))
        ann3 = json.load(open(os.path.join('/data/Datasets/Vision-Language','alpaca_gpt4_data_zh.json')))
        ann4 = json.load(open(os.path.join('/data/Datasets/Vision-Language','alpaca_data_zh_51k.json')))
        self.ann = ann1 + ann2 + ann3 + ann4
        self.lang_type = ['EN'] * len(ann1) + ['EN'] * len(ann2) + ['CH'] * len(ann3) + ['CH'] * len(ann4)
        print(f"total length: {len(self)}")
        self.transform = transform_train
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            # filename = data_item['image'].replace('/data0/data', '/mnt/petrelfs/share_data/hanjiaming')


            filename = os.path.join("/data/Datasets/COCO/train2014","COCO_train2014_" + data_item['image'])
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']


            # < fill path substitution logics here>
            # filename = url.replace("/data0/data/coco/", "/mnt/petrelfs/leimeng/datasets/coco/")

            image = Image.open(filename).convert('RGB')
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 336, 336)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input, self.lang_type[index])
        input2 = input1 + answer
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