"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
import zss
from datasets import load_dataset
from nltk import edit_distance
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from zss import Node

import logging
# from math import prod
from pathlib import Path
from functools import partial
import random
from PIL import Image, UnidentifiedImageError


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)


class Img2TableDatasetTest2(Dataset):
    def __init__(self, 
                 split = 'test'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_arxiv = []
        self.data_plotQA = []
        self.data_chartQA = []
        self.data_pie = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/annotations3.json'
            json_file2 = '/mnt/petrelfs/plotQA/train/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Imgs_all_onlygentable1.json'
            json_file4 = '/mnt/petrelfs/other/train.json'
            json_file5 = '/mnt/petrelfs/plotQA2/train1.json'
            json_file11 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/annotations4.json'
            json_file21 = '/mnt/petrelfs/plotQA/val/annotations3.json'
            json_file41 = '/mnt/petrelfs/other/val.json'
            json_file51 = '/mnt/petrelfs/plotQA2/val1.json'
            json_file6 = '/mnt/petrelfs/plotQA3/train.json'
            json_file61 = '/mnt/petrelfs/plotQA3/val.json'
            json_file7 = '/mnt/petrelfs/plotQA4/train.json'
            json_file71 = '/mnt/petrelfs/plotQA4/val.json'
            json_file9 = '/mnt/petrelfs/plotQA5/train.json'
            json_file91 = '/mnt/petrelfs/plotQA5/val.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data_arxiv = json.load(f3)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file6, 'r') as f6:
                self.data6 = json.load(f6)
            with open(json_file7, 'r') as f7:
                self.data7 = json.load(f7)
            with open(json_file9, 'r') as f9:
                self.data9 = json.load(f9)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file61, 'r') as f61:
                self.data61 = json.load(f61)
            with open(json_file71, 'r') as f71:
                self.data71 = json.load(f71)
            with open(json_file91, 'r') as f91:
                self.data91 = json.load(f91)

            self.data1 = self.data1 + self.data11
            self.data2 = self.data2 + self.data21
            self.data4 = self.data4 + self.data41
            self.data5 = self.data5 + self.data51
            self.data6 = self.data6 + self.data61
            self.data7 = self.data7 + self.data71
            self.data9 = self.data9 + self.data91
            self.data_plotQA = self.data2 + self.data5 + self.data6 + self.data7 + self.data9
            self.data_chartQA = self.data1
            self.data_pie = self.data4
            
            # self.data = self.data_arxiv + self.data_plotQA + self.data_chartQA + self.data_pie
            # self.data = self.data[:100]
        else:  # test or val
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/annotations2.json'
            json_file2 = '/mnt/petrelfs/plotQA/test/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/test.json'
            json_file4 = '/mnt/petrelfs/other/test.json'
            with open(json_file2, 'r') as f2:
                self.data = json.load(f2)

    def __len__(self):
        # return len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5
        return len(self.data)

    def __getitem__(self, index):
        # entry = self.data[index]
        # entry = self.data[index]
        entry = {
                    'image': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_arxiv, self.data_chartQA, self.data_pie, self.data_plotQA]
            weights = [5, 6, 1, 6]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == "quanfeng_synthetic":
                img_folder = '/mnt/petrelfs/share_data/dataset/synthetic/img/' + entry['type']
            elif entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA2_train":
                img_folder = '/mnt/petrelfs/plotQA2/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA3_train":
                img_folder = '/mnt/petrelfs/plotQA3/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA4_train":
                img_folder = '/mnt/petrelfs/plotQA4/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA5_train":
                img_folder = '/mnt/petrelfs/share_data/dataset/plotQA3_train/' + entry['type']
            elif entry['datafrom'] == "chartQA_train":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
            elif entry['datafrom'] == "plotQA2_val":
                img_folder = '/mnt/petrelfs/plotQA2/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA3_val":
                img_folder = '/mnt/petrelfs/plotQA3/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA4_val":
                img_folder = '/mnt/petrelfs/plotQA4/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA5_val":
                img_folder = '/mnt/petrelfs/share_data/dataset/val/' + entry['type']
            elif entry['datafrom'] == "chartQA_val":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'arxiv':
                img_folder = ''   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            elif entry['datafrom'] == 'chartQA_test':
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        table = entry["table"]
        img_path = os.path.join(img_folder,entry['img'])
        if not os.path.exists(img_path) or table==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        try:
            img = Image.open(img_path)
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        sample = {
            'image': img,
            'ground_truth': table
        }
        return sample


class Img2TableDatasetTest1(Dataset):
    def __init__(self, 
                 split = 'test'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_arxiv = []
        self.data_plotQA = []
        self.data_chartQA = []
        self.data_pie = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/annotations3.json'
            json_file2 = '/mnt/petrelfs/plotQA/train/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Imgs_all_onlygentable1.json'
            json_file4 = '/mnt/petrelfs/other/train.json'
            json_file5 = '/mnt/petrelfs/plotQA2/train1.json'
            json_file11 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/annotations4.json'
            json_file21 = '/mnt/petrelfs/plotQA/val/annotations3.json'
            json_file41 = '/mnt/petrelfs/other/val.json'
            json_file51 = '/mnt/petrelfs/plotQA2/val1.json'
            json_file6 = '/mnt/petrelfs/plotQA3/train.json'
            json_file61 = '/mnt/petrelfs/plotQA3/val.json'
            json_file7 = '/mnt/petrelfs/plotQA4/train.json'
            json_file71 = '/mnt/petrelfs/plotQA4/val.json'
            json_file9 = '/mnt/petrelfs/plotQA5/train.json'
            json_file91 = '/mnt/petrelfs/plotQA5/val.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data_arxiv = json.load(f3)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file6, 'r') as f6:
                self.data6 = json.load(f6)
            with open(json_file7, 'r') as f7:
                self.data7 = json.load(f7)
            with open(json_file9, 'r') as f9:
                self.data9 = json.load(f9)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file61, 'r') as f61:
                self.data61 = json.load(f61)
            with open(json_file71, 'r') as f71:
                self.data71 = json.load(f71)
            with open(json_file91, 'r') as f91:
                self.data91 = json.load(f91)

            self.data1 = self.data1 + self.data11
            self.data2 = self.data2 + self.data21
            self.data4 = self.data4 + self.data41
            self.data5 = self.data5 + self.data51
            self.data6 = self.data6 + self.data61
            self.data7 = self.data7 + self.data71
            self.data9 = self.data9 + self.data91
            self.data_plotQA = self.data2 + self.data5 + self.data6 + self.data7 + self.data9
            self.data_chartQA = self.data1
            self.data_pie = self.data4
            
            # self.data = self.data_arxiv + self.data_plotQA + self.data_chartQA + self.data_pie
            # self.data = self.data[:100]
        else:  # test or val
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/annotations2.json'
            json_file2 = '/mnt/petrelfs/plotQA/test/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/test.json'
            json_file4 = '/mnt/petrelfs/other/test.json'
            with open(json_file1, 'r') as f1:
                self.data = json.load(f1)

    def __len__(self):
        # return len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5
        return len(self.data)

    def __getitem__(self, index):
        # entry = self.data[index]
        # entry = self.data[index]
        entry = {
                    'image': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_arxiv, self.data_chartQA, self.data_pie, self.data_plotQA]
            weights = [5, 6, 1, 6]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == "quanfeng_synthetic":
                img_folder = '/mnt/petrelfs/share_data/dataset/synthetic/img/' + entry['type']
            elif entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA2_train":
                img_folder = '/mnt/petrelfs/plotQA2/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA3_train":
                img_folder = '/mnt/petrelfs/plotQA3/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA4_train":
                img_folder = '/mnt/petrelfs/plotQA4/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA5_train":
                img_folder = '/mnt/petrelfs/share_data/dataset/plotQA3_train/' + entry['type']
            elif entry['datafrom'] == "chartQA_train":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
            elif entry['datafrom'] == "plotQA2_val":
                img_folder = '/mnt/petrelfs/plotQA2/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA3_val":
                img_folder = '/mnt/petrelfs/plotQA3/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA4_val":
                img_folder = '/mnt/petrelfs/plotQA4/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA5_val":
                img_folder = '/mnt/petrelfs/share_data/dataset/val/' + entry['type']
            elif entry['datafrom'] == "chartQA_val":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'arxiv':
                img_folder = ''   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            elif entry['datafrom'] == 'chartQA_test':
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        table = entry["table"]
        img_path = os.path.join(img_folder,entry['img'])
        if not os.path.exists(img_path) or table==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        try:
            img = Image.open(img_path)
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        sample = {
            'image': img,
            'ground_truth': table
        }
        return sample



class Img2TableDatasetTest(Dataset):
    def __init__(self, 
                 split = 'test'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_arxiv = []
        self.data_plotQA = []
        self.data_chartQA = []
        self.data_pie = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/annotations3.json'
            json_file2 = '/mnt/petrelfs/plotQA/train/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Imgs_all_onlygentable1.json'
            json_file4 = '/mnt/petrelfs/other/train.json'
            json_file5 = '/mnt/petrelfs/plotQA2/train1.json'
            json_file11 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/annotations4.json'
            json_file21 = '/mnt/petrelfs/plotQA/val/annotations3.json'
            json_file41 = '/mnt/petrelfs/other/val.json'
            json_file51 = '/mnt/petrelfs/plotQA2/val1.json'
            json_file6 = '/mnt/petrelfs/plotQA3/train.json'
            json_file61 = '/mnt/petrelfs/plotQA3/val.json'
            json_file7 = '/mnt/petrelfs/plotQA4/train.json'
            json_file71 = '/mnt/petrelfs/plotQA4/val.json'
            json_file9 = '/mnt/petrelfs/plotQA5/train.json'
            json_file91 = '/mnt/petrelfs/plotQA5/val.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data_arxiv = json.load(f3)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file6, 'r') as f6:
                self.data6 = json.load(f6)
            with open(json_file7, 'r') as f7:
                self.data7 = json.load(f7)
            with open(json_file9, 'r') as f9:
                self.data9 = json.load(f9)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file61, 'r') as f61:
                self.data61 = json.load(f61)
            with open(json_file71, 'r') as f71:
                self.data71 = json.load(f71)
            with open(json_file91, 'r') as f91:
                self.data91 = json.load(f91)

            self.data1 = self.data1 + self.data11
            self.data2 = self.data2 + self.data21
            self.data4 = self.data4 + self.data41
            self.data5 = self.data5 + self.data51
            self.data6 = self.data6 + self.data61
            self.data7 = self.data7 + self.data71
            self.data9 = self.data9 + self.data91
            self.data_plotQA = self.data2 + self.data5 + self.data6 + self.data7 + self.data9
            self.data_chartQA = self.data1
            self.data_pie = self.data4
            
            # self.data = self.data_arxiv + self.data_plotQA + self.data_chartQA + self.data_pie
            # self.data = self.data[:100]
        else:  # test or val
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/annotations2.json'
            json_file2 = '/mnt/petrelfs/plotQA/test/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/test.json'
            json_file4 = '/mnt/petrelfs/other/test.json'
            with open(json_file1, 'r') as f1:
                self.data = json.load(f1)

    def __len__(self):
        # return len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5
        return len(self.data)

    def __getitem__(self, index):
        task = '<extract_data_table>'
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_arxiv, self.data_chartQA, self.data_pie, self.data_plotQA]
            weights = [5, 6, 1, 6]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == "quanfeng_synthetic":
                img_folder = '/mnt/petrelfs/share_data/dataset/synthetic/img/' + entry['type']
            elif entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA2_train":
                img_folder = '/mnt/petrelfs/plotQA2/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA3_train":
                img_folder = '/mnt/petrelfs/plotQA3/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA4_train":
                img_folder = '/mnt/petrelfs/plotQA4/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA5_train":
                img_folder = '/mnt/petrelfs/share_data/dataset/plotQA3_train/' + entry['type']
            elif entry['datafrom'] == "chartQA_train":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
            elif entry['datafrom'] == "plotQA2_val":
                img_folder = '/mnt/petrelfs/plotQA2/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA3_val":
                img_folder = '/mnt/petrelfs/plotQA3/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA4_val":
                img_folder = '/mnt/petrelfs/plotQA4/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA5_val":
                img_folder = '/mnt/petrelfs/share_data/dataset/val/' + entry['type']
            elif entry['datafrom'] == "chartQA_val":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'arxiv':
                img_folder = ''   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            elif entry['datafrom'] == 'chartQA_test':
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'image': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        table = entry["table"]
        img_path = os.path.join(img_folder,entry['img'])
        if not os.path.exists(img_path) or table==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        try:
            img = Image.open(img_path)
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                'image': None,
                'ground_truth': None
            }
            return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
        sample = {
            'image': img,
            'ground_truth': table
        }
        return sample


class Img2TableDataset(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_arxiv = []
        self.data_plotQA = []
        self.data_chartQA = []
        self.data_pie = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/annotations3.json'
            json_file2 = '/mnt/petrelfs/plotQA/train/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Imgs_all_onlygentable1.json'
            json_file4 = '/mnt/petrelfs/other/train.json'
            json_file5 = '/mnt/petrelfs/plotQA2/train1.json'
            json_file11 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/annotations4.json'
            json_file21 = '/mnt/petrelfs/plotQA/val/annotations3.json'
            json_file41 = '/mnt/petrelfs/other/val.json'
            json_file51 = '/mnt/petrelfs/plotQA2/val1.json'
            json_file6 = '/mnt/petrelfs/plotQA3/train.json'
            json_file61 = '/mnt/petrelfs/plotQA3/val.json'
            json_file7 = '/mnt/petrelfs/plotQA4/train.json'
            json_file71 = '/mnt/petrelfs/plotQA4/val.json'
            json_file9 = '/mnt/petrelfs/plotQA5/train.json'
            json_file91 = '/mnt/petrelfs/plotQA5/val.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data_arxiv = json.load(f3)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file6, 'r') as f6:
                self.data6 = json.load(f6)
            with open(json_file7, 'r') as f7:
                self.data7 = json.load(f7)
            with open(json_file9, 'r') as f9:
                self.data9 = json.load(f9)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file61, 'r') as f61:
                self.data61 = json.load(f61)
            with open(json_file71, 'r') as f71:
                self.data71 = json.load(f71)
            with open(json_file91, 'r') as f91:
                self.data91 = json.load(f91)

            self.data1 = self.data1 + self.data11
            self.data2 = self.data2 + self.data21
            self.data4 = self.data4 + self.data41
            self.data5 = self.data5 + self.data51
            self.data6 = self.data6 + self.data61
            self.data7 = self.data7 + self.data71
            self.data9 = self.data9 + self.data91
            self.data_plotQA = self.data2 + self.data5 + self.data6 + self.data7 + self.data9
            self.data_chartQA = self.data1
            self.data_pie = self.data4
            
            self.data = self.data_arxiv + self.data_plotQA + self.data_chartQA + self.data_pie
            # self.data = self.data[:100]
        else:  # test or val
            json_file1 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/annotations2.json'
            json_file2 = '/mnt/petrelfs/plotQA/test/annotations3.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/test.json'
            json_file4 = '/mnt/petrelfs/other/test.json'
            with open(json_file2, 'r') as f2:
                self.data = json.load(f2)
            self.data = self.data[:100]

    def __len__(self):
        return len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5
        # return len(self.data)
    def __getitem__(self, index):
        task = '<extract_data_table>'
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_arxiv, self.data_chartQA, self.data_pie, self.data_plotQA]
            weights = [3, 6, 2, 6]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == "quanfeng_synthetic":
                img_folder = '/mnt/petrelfs/share_data/dataset/synthetic/img/' + entry['type']
            elif entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA2_train":
                img_folder = '/mnt/petrelfs/plotQA2/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA3_train":
                img_folder = '/mnt/petrelfs/plotQA3/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA4_train":
                img_folder = '/mnt/petrelfs/plotQA4/Img/train/' + entry['type']
            elif entry['datafrom'] == "plotQA5_train":
                img_folder = '/mnt/petrelfs/share_data/dataset/plotQA3_train/' + entry['type']
            elif entry['datafrom'] == "chartQA_train":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
            elif entry['datafrom'] == "plotQA2_val":
                img_folder = '/mnt/petrelfs/plotQA2/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA3_val":
                img_folder = '/mnt/petrelfs/plotQA3/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA4_val":
                img_folder = '/mnt/petrelfs/plotQA4/Img/val/' + entry['type']
            elif entry['datafrom'] == "plotQA5_val":
                img_folder = '/mnt/petrelfs/share_data/dataset/val/' + entry['type']
            elif entry['datafrom'] == "chartQA_val":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'arxiv':
                img_folder = ''   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            elif entry['datafrom'] == 'chartQA_test':
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 

        table = entry["table"]
        # print(table)
        # img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
        img_path = os.path.join(img_folder,entry['img'])
        # print(img_path)
        if not os.path.exists(img_path) or table==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                'task': None,
                'image': None,
                'query': None,
                'ground_truth': None
            }
            if self.split == 'train':
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
            
            else:
                return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                'task': None,
                'image': None,
                'query': None,
                'ground_truth': None
            }
            if self.split == 'train':
                return self.__getitem__((index + 1)%(len(self.data_arxiv) + len(self.data_chartQA) + len(self.data_pie) + len(self.data_plotQA) // 5)) 
            
            else:
                return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': task,
            'image': img,
            'query': "Extract the table corresponding to this chart.",
            'ground_truth': table
        }
        return sample



class ChartSummDataset(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_opencqa = []
        self.data_vistext = []
        self.data_chartsumm = []
        self.data_chart2text = []
        self.data_charttotext = []
        self.data_unichart = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/openCQA/train_onlysumm.json'
            json_file2 = '/mnt/petrelfs/vistext/data/train.json'
            json_file3 = '/mnt/petrelfs/ChartSumm/train_ks.json'
            json_file4 = '/mnt/petrelfs/Chart2Text/train.json'
            json_file5 = '/mnt/petrelfs/Chart-to-text/pew/trainval.json'
            json_file6 = '/mnt/petrelfs/share_data/mengfanqing1/Unichart_Data/chartsumm.json'
            json_file11 = '/mnt/petrelfs/openCQA/val_onlysumm.json'
            json_file21 = '/mnt/petrelfs/vistext/data/train.json'
            json_file31 = '/mnt/petrelfs/ChartSumm/val_ks.json'
            json_file41 = '/mnt/petrelfs/Chart2Text/valid.json'
            json_file51 = '/mnt/petrelfs/Chart-to-text/statista/trainval.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data3 = json.load(f3)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file6, 'r') as f6:
                self.data_unichart = json.load(f6)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)
            with open(json_file31, 'r') as f31:
                self.data31 = json.load(f31)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)

            self.data_opencqa = self.data1 + self.data11
            self.data_vistext = self.data2 + self.data21
            self.data_chartsumm = self.data3 + self.data31
            self.data_chart2text = self.data4 + self.data41
            self.data_charttotext = self.data5 + self.data51
            self.data = self.data_opencqa + self.data_vistext + self.data_chartsumm + self.data_chart2text + self.data_charttotext
            # self.data = self.data[:100]

        else:  # test or val
            json_file1 = '/mnt/petrelfs/Chart-to-text/pew/test.json'
            json_file2 = '/mnt/petrelfs/Chart-to-text/statista/test.json'
            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            self.data = self.data1[:100]

    def __len__(self):
        return len(self.data_opencqa) + len(self.data_vistext) + len(self.data_chartsumm) + len(self.data_chart2text) + len(self.data_charttotext) + len(self.data_unichart)

    def __getitem__(self, index):
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_opencqa, self.data_vistext, self.data_chartsumm, self.data_chart2text, self.data_charttotext, self.data_unichart]
            weights = [1, 1, 1, 1, 1, 1]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "opencqa":
                img_folder = '/mnt/petrelfs/share_data/OpenCQA/chart_images'
            elif entry['datafrom'] == "vistext":
                img_folder = '/mnt/petrelfs/vistext/data/images'
            elif entry['datafrom'] == "chartsumm":
                img_folder = '/mnt/petrelfs/share_data/dataset/ChartSummData/ChartImages'
            elif entry['datafrom'] == "chart2text":
                if 'single' in entry['type']:
                    img_folder = '/mnt/petrelfs/share_data/Chart2Text/Chart2TextImages/images/statista'
                elif 'multicolumn' in entry['type']:
                    img_folder = '/mnt/petrelfs/share_data/Chart2Text/Chart2TextImages/multiColumn/images/statista'
            elif entry['datafrom'] == "chart-to-text_pew":
                img_folder = '/mnt/petrelfs/share_data/Chart-to-text/pew_dataset/dataset'
            elif entry['datafrom'] == "chart-to-text_statista":
                img_folder = '/mnt/petrelfs/share_data/Chart-to-text/statista_dataset/dataset'
            elif entry['datafrom'] == "unichart":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/Unichart_Data/Img/content/Images'
           
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data_opencqa)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'arxiv':
                img_folder = ''   # path in it
            elif entry['datafrom'] == "other":
                img_folder = '/mnt/petrelfs/other/png'
            elif entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            elif entry['datafrom'] == 'chartQA_test':
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 


        if "unichart" in entry['datafrom']:
            imgname = entry['imgname']
        else:
            imgname = entry['img']
        
        if "opencqa" in entry['datafrom']:
            summ = entry['summary']
        elif "unichart" in entry['datafrom']:
            summ = entry['label']
        else:
            summ = entry['summ']

        # opencqa img summary
        # vistext img summ
        # unichart imgname label
        # chartsumm img summ
        # chart2text img summ
        # chart-to-text img summ
        # print(table)
        # img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
        img_path = os.path.join(img_folder,imgname)
        # print(img_path)
        if not os.path.exists(img_path) or summ==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': '<summarize_chart>',
            'image': img,
            'query': "Summarize the content of this chart.",
            'ground_truth': summ
        }
        return sample


class ChartOpenQADataset(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        self.data_opencqa = []
        self.data_arxiv = []
        self.data_scigraph = []
        self.data_chartqa = []
        self.data_plotqa = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/openCQA/train_onlyqa.json'
            json_file2 = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/train_qa.json'
            json_file3 = '/mnt/petrelfs/share_data/mengfanqing1/ScigraphQA/train_onlyQA.json'
            json_file4 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/train_qa_aug_human.json'
            json_file41 = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/val_qa_aug_human.json'
            json_file5 = '/mnt/petrelfs/plotQA/QA/train_else_onlyqa.json'
            json_file51 = '/mnt/petrelfs/plotQA/QA/val_else_onlyqa.json'
            json_file11 = '/mnt/petrelfs/openCQA/val_onlyqa.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file3, 'r') as f3:
                self.data3 = json.load(f3)
            with open(json_file4, 'r') as f4:
                self.data4 = json.load(f4)
            with open(json_file5, 'r') as f5:
                self.data5 = json.load(f5)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file41, 'r') as f41:
                self.data41 = json.load(f41)
            with open(json_file51, 'r') as f51:
                self.data51 = json.load(f51)

            self.data_opencqa = self.data1 + self.data11
            self.data_chartqa = self.data4 + self.data41
            self.data_plotqa = self.data5 + self.data51
            self.data_arxiv = self.data2
            self.data_scigraph = self.data3
            self.data = self.data_opencqa + self.data_arxiv + self.data_scigraph + self.data_chartqa + self.data_plotqa
            # self.data = self.data[:100]
            

        else:  # test or val
            json_file1 = '/mnt/petrelfs/openCQA/test_onlyqa.json'
            with open(json_file1, 'r') as f1:
                self.data = json.load(f1)
            self.data = self.data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            # chartQA, plotQA, other(pie + synthetic+ plotQA5), plotQA2, plotQA3, plotQA4, plotQA5
            choices = [self.data_opencqa, self.data_arxiv, self.data_scigraph, self.data_chartqa, self.data_plotqa]
            weights = [3, 1, 1, 4, 2]
            data = random.choices(choices, weights=weights)[0]
            entry = data[index % len(data)]
            if entry['datafrom'] == "opencqa":
                img_folder = '/mnt/petrelfs/share_data/OpenCQA/chart_images'
            elif entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'
            elif entry['datafrom'] == "scigraph":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/ScigraphQA/imgs/train'
            elif entry['datafrom'] == "chartQA_train":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/train/png'
            elif entry['datafrom'] == "chartQA_val":
                img_folder = '/mnt/petrelfs/ChartQA-main/ChartQA-Dataset/val/png'
            elif entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == "opencqa":
                img_folder = '/mnt/petrelfs/share_data/OpenCQA/chart_images'
            elif entry['datafrom'] == "arxiv":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/arxiv/Img_QA'
            elif entry['datafrom'] == "scigraph":
                img_folder = '/mnt/petrelfs/share_data/mengfanqing1/ScigraphQA/imgs/test'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 


        imgname = entry["img"]
        question = entry["question"]
        answer = entry["answer"]
        # opencqa img question answer
        # arxiv img question answer
        # scigraph img question answer

        img_path = os.path.join(img_folder,imgname)
        # print(img_path)
        if not os.path.exists(img_path) or question==None or answer==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': '<opencqa>',
            'image': img,
            'query': question,
            'ground_truth': answer
        }
        return sample



class ChartMathQADataset(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/plotQA/QA/train_commandline_onlyqa_json2token_nofunclist.json'
            json_file11 = '/mnt/petrelfs/plotQA/QA/val_commandline_onlyqa_json2token_nofunclist.json'
            json_file2 = '/mnt/petrelfs/plotQA/extraQA/qa_pairs_commandline_tiny_train_token.json'
            json_file21 = '/mnt/petrelfs/plotQA/extraQA/qa_pairs_commandline_tiny_vali_token.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            # with open(json_file2, 'r') as f2:
            #     self.data2 = json.load(f2)
            # with open(json_file21, 'r') as f21:
            #     self.data21 = json.load(f21)

            # self.data = random.sample(self.data1, k=len(self.data1)//4) + random.sample(self.data11, k=len(self.data11)//4) + self.data2 + self.data21
            # self.data = self.data11[:1000]
            self.data = self.data1 + self.data11
            # self.data = self.data1 + self.data11 + self.data2 + self.data21
            # self.data = random.sample(self.data,k = (len(self.data)*3) // 4)
            # self.data = self.data[:10000]
            # self.data = self.data
            # self.data = random.sample(self.data,k = (len(self.data) // 2))
            

        else:  # test or val
            json_file11 = '/mnt/petrelfs/plotQA/QA/val_commandline_onlyqa_json2token_nofunclist.json'
            with open(json_file11, 'r') as f1:
                self.data = json.load(f1)
            self.data = self.data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            entry = self.data[index % len(self.data)]
            if entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
           
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 


        imgname = entry["img"]
        question = entry["question"]
        if "answer_commandline" in entry.keys():
            answer = entry["answer_commandline"]
        elif "commandline" in entry.keys():
            answer = entry["commandline"]
        else:
            print("Sample could not be found.")
            return self.__getitem__((index + 1)%len(self.data)) 
        # opencqa img question answer
        # arxiv img question answer
        # scigraph img question answer

        img_path = os.path.join(img_folder,imgname)
        # print(img_path)
        if not os.path.exists(img_path) or question==None or answer==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': '<mathqa>',
            'image': img,
            'query': question,
            'ground_truth': answer
        }
        return sample




class ChartMathQADataset1(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/plotQA/QA/train_commandline_onlyqa_json2token_nofunclist.json'
            json_file11 = '/mnt/petrelfs/plotQA/QA/val_commandline_onlyqa_json2token_nofunclist.json'
            json_file2 = '/mnt/petrelfs/plotQA/extraQA/qa_pairs_commandline_tiny_train_token.json'
            json_file21 = '/mnt/petrelfs/plotQA/extraQA/qa_pairs_commandline_tiny_vali_token.json'

            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)

            # self.data = random.sample(self.data1, k=len(self.data1)//4) + random.sample(self.data11, k=len(self.data11)//4) + self.data2 + self.data21
            # self.data = self.data11[:1000]
            self.data = self.data2 + self.data21
            # self.data = random.sample(self.data,k = (len(self.data)*3) // 4)
            # self.data = self.data[:10000]
            # self.data = self.data
            # self.data = random.sample(self.data,k = (len(self.data) // 2))
            

        else:  # test or val
            json_file11 = '/mnt/petrelfs/plotQA/QA/val_commandline_onlyqa_json2token_nofunclist.json'
            with open(json_file11, 'r') as f1:
                self.data = json.load(f1)
            self.data = self.data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            entry = self.data[index % len(self.data)]
            if entry['datafrom'] == "plotQA_train":
                img_folder = '/mnt/petrelfs/plotQA/train/png/png'
            elif entry['datafrom'] == "plotQA_val":
                img_folder = '/mnt/petrelfs/plotQA/val/png'
           
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 


        imgname = entry["img"]
        question = entry["question"]
        if "answer_commandline" in entry.keys():
            answer = entry["answer_commandline"]
        elif "commandline" in entry.keys():
            answer = entry["commandline"]
        else:
            print("Sample could not be found.")
            return self.__getitem__((index + 1)%len(self.data)) 
        # opencqa img question answer
        # arxiv img question answer
        # scigraph img question answer

        img_path = os.path.join(img_folder,imgname)
        # print(img_path)
        if not os.path.exists(img_path) or question==None or answer==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': '<mathqa>',
            'image': img,
            'query': question,
            'ground_truth': answer
        }
        return sample


class ChartReferringQADataset(Dataset):
    def __init__(self, 
                 split = 'train'):
        # json_file = os.path.join(self.root_dir,split+'.jsonl')
        self.split = split
        self.data = []
        if self.split == 'train':
            json_file1 = '/mnt/petrelfs/plotQA/referring_QA/plotQA_referring_box_QA_commandline_train_token.json'
            json_file11 = '/mnt/petrelfs/plotQA/referring_QA/plotQA_referring_box_QA_commandline_val_token.json'
            json_file2 = '/mnt/petrelfs/plotQA/referring_QA/train_plotQA_referring_box_QA.json'
            json_file21 = '/mnt/petrelfs/plotQA/referring_QA/val_plotQA_referring_box_QA.json'

            with open(json_file1, 'r') as f1:
                self.data1 = json.load(f1)
            with open(json_file11, 'r') as f11:
                self.data11 = json.load(f11)
            with open(json_file2, 'r') as f2:
                self.data2 = json.load(f2)
            with open(json_file21, 'r') as f21:
                self.data21 = json.load(f21)

            self.data = self.data1 + self.data11 + self.data2 + self.data21
            # self.data = self.data[:100]
            

        else:  # test or val
            json_file1 = '/mnt/petrelfs/plotQA/referring_QA/val_plotQA_referring_box_QA_commandline.json'
            with open(json_file1, 'r') as f1:
                self.data = json.load(f1)
            # self.data = self.data[:100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
        if self.split == 'train':
            entry = self.data[index % len(self.data)]
            if entry['datafrom'] == "referringplotQA_train":
                img_folder = '/mnt/petrelfs/share_data/referring_box/train/img'
            elif entry['datafrom'] == "referringplotQA_val":
                img_folder = '/mnt/petrelfs/share_data/referring_box/val/img'
           
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 

        else:  # test
            entry = self.data[index]
            if entry['datafrom'] == 'plotQA_test':
                img_folder = '/mnt/petrelfs/plotQA/test/png'
            else:
                print("Sample %s could not be found.")
                sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
                return self.__getitem__((index + 1)%len(self.data)) 


        imgname = entry["img"]
        question = entry["question"]
        if "commandline" in entry.keys():
            answer = entry["commandline"]
        else:
            answer = "<s_question>" + str(question) + "</s_question>" + "<s_answer>" + str(entry["answer"]) + "</s_answer>"
        # opencqa img question answer
        # arxiv img question answer
        # scigraph img question answer

        img_path = os.path.join(img_folder,imgname)
        # print(img_path)
        if not os.path.exists(img_path) or question==None or answer==None:
            print("Sample %s could not be found.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print("Image %s could not be opened.", img_path)
            sample = {
                    'task': None,
                    'image': None,
                    'query': None,
                    'ground_truth': None
                }
            return self.__getitem__((index + 1)%len(self.data)) 
        sample = {
            'task': '<referringqa>',
            'image': img,
            'query': '<referringqa>' + question,
            'ground_truth': answer
        }
        return sample




class DonutMultiTaskDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset_ocr = Img2TableDataset(
            split=self.split
        )
        self.dataset_summ = ChartSummDataset(
            split=self.split
        )
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.dataset_opencqa = ChartOpenQADataset(
            split=self.split
        )
        self.dataset_referringqa = ChartReferringQADataset(
            split=self.split
        )
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = max(len(self.dataset_ocr),len(self.dataset_summ),len(self.dataset_mathqa),len(self.dataset_opencqa),len(self.dataset_referringqa))
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        choices = ['ocr','summ','mathqa','opencqa','referringqa']
        # choices = [self.dataset_ocr, self.dataset_summ, self.dataset_mathqa, self.dataset_opencqa]
        weights = [1, 3, 5, 5, 6]
        choose = random.choices(choices, weights=weights)[0]
        if 'ocr' in choose:
            data = self.dataset_ocr
        elif 'summ' in choose:
            data = self.dataset_summ
        elif 'mathqa' in choose:
            data = self.dataset_mathqa
        elif 'opencqa' in choose:
            data = self.dataset_opencqa
        elif 'referringqa' in choose:
            data = self.dataset_referringqa
        else:
            data = self.dataset_ocr
        # data = self.dataset_summ
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']
        # input_tensor
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        # input_ids
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.donut_model.decoder.tokenizer.eos_token 
        try:
            # print(sample)
            processed_parse = str(task_prefix) + " " + str(sample['query']) + " " + self.prompt_end_token + " " + str(sample['ground_truth']) + self.donut_model.decoder.tokenizer.eos_token 
        except:
            print('error!!!',task_prefix, sample)
            assert 0
            return
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse

class DonutOriMathQADataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.prompt_extra = "The provided function list is: ['numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum', 'numpy.add', 'numpy.subtract','numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=', 'numpy.==', 'numpy.!=','numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt', 'numpy.square', 'numpy.log','numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any', 'numpy.diff', 'numpy.corrcoef','numpy.cov','max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', 'getitem', 'select', '<', '<=', '>', '>=', '==', '!=']. You need to output in this format: <s_step1><s_func1>the function name you choose, must be in the provided function list</s_func1><s_arg1>the parameter list of the func1</s_arg1><s_output1>the variable name output by the func1</s_output1></s_step1><s_step2><s_func2>the function name you choose, must be in the provided function list</s_func2><s_arg2>the parameter list of the func2</s_arg2><s_output2>the variable name output by the func2</s_output2></s_step2>... Question: "
        
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = len(self.dataset_mathqa)
       
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.gt_token_sequences = []
        self.gt_token_sequences_idx_forimg = []


    def __len__(self) -> int:
        return self.dataset_length
        # return len(self.gt_token_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        
        data = self.dataset_mathqa
        
        # idx = self.gt_token_sequences_idx_forimg[index]
        sample = data[index % len(data)]
        ground_truth = sample['ground_truth']
        # sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")

        # input_ids
        # processed_parse = self.gt_token_sequences[index]
        processed_parse = self.task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
                    
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse



class DonutMathQADataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.prompt_extra = "The provided function list is: ['numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum', 'numpy.add', 'numpy.subtract','numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=', 'numpy.==', 'numpy.!=','numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt', 'numpy.square', 'numpy.log','numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any', 'numpy.diff', 'numpy.corrcoef','numpy.cov','max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', 'getitem', '<', '<=', '>', '>=', '==', '!=']. You need to output in this format: {'step1':{'func1': the function name you choose, must be in the provided function list, 'arg1': the parameter list of the func1 ,'output1': the variable name output by the func1}, 'step2': {'func2': the function name you choose, must be in the provided  function list, 'args2': the parameter list of the func2 ,'output2': the variable name output by the func2}, ...}."
        
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = len(self.dataset_mathqa)
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        
        data = self.dataset_mathqa
        
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']
        # input_tensor
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        # input_ids
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.donut_model.decoder.tokenizer.eos_token 
        try:
            # print(sample)
            # processed_parse = str(task_prefix) + " " + str(sample['query']) + " " + self.prompt_extra + " " + self.prompt_end_token + " " + str(sample['ground_truth']) + self.donut_model.decoder.tokenizer.eos_token 
            processed_parse = str(task_prefix) + " " + str(sample['query']) + " " + self.prompt_extra + " " + self.prompt_end_token + " " + str(sample['ground_truth']) + self.donut_model.decoder.tokenizer.eos_token 
        except:
            print('error!!!',task_prefix, sample)
            assert 0
            return
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse

class DonutDataset2(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        task_prefix: str = '<chartqa>',
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset = self.dataset = Img2TableDataset(
            split=self.split
        )
        #list(dict)
        self.dataset_length = len(self.dataset)
        self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        # image_path = os.path.join(self.img_folder, sample['img'])
        image = sample['image']
        # input_tensor
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        # input_ids
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.donut_model.decoder.tokenizer.eos_token 
        processed_parse = self.task_prefix + " " + "<extract_data_table>" + " " + self.prompt_end_token + " " + sample['ground_truth'] + self.donut_model.decoder.tokenizer.eos_token 
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse



class DonutDataset2Test(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        task_prefix: str = '<chartqa>',
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset = self.dataset = Img2TableDatasetTest(
            split=self.split
        )
        #list(dict)
        self.dataset_length = len(self.dataset)
        self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        # image_path = os.path.join(self.img_folder, sample['img'])
        image = sample['image']
        # input_tensor
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        # input_ids
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.donut_model.decoder.tokenizer.eos_token 
        processed_parse = self.task_prefix + " " + "<extract_data_table>" + " " + self.prompt_end_token + " " + sample['ground_truth'] + self.donut_model.decoder.tokenizer.eos_token 
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class DonutDataset1(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        img_folder: str,
        donut_model: PreTrainedModel,
        max_length: int,
        task_prefix: str = '<chartqa>',
        split: str = "train",
        ignore_id: int = -100,
        # task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        # self.task_start_token = task_start_token
        # self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.img_folder = img_folder
        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset = load_json(dataset_name_or_path)[:100]
        self.dataset_length = len(self.dataset)
        self.task_prefix = task_prefix
        # self.gt_token_sequences = []
        # for sample in self.dataset:
        #     ground_truth = json.loads(sample["ground_truth"])
        #     if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
        #         assert isinstance(ground_truth["gt_parses"], list)
        #         gt_jsons = ground_truth["gt_parses"]
        #     else:
        #         assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
        #         gt_jsons = [ground_truth["gt_parse"]]

        #     self.gt_token_sequences.append(
        #         [
        #             task_start_token
        #             + self.donut_model.json2token(
        #                 gt_json,
        #                 update_special_tokens_for_json_key=self.split == "train",
        #                 sort_json_key=self.sort_json_key,
        #             )
        #             + self.donut_model.decoder.tokenizer.eos_token
        #             for gt_json in gt_jsons  # load json from list of json
        #         ]
        #     )

        # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]
        image_path = os.path.join(self.img_folder, sample['img'])
        # input_tensor
        
        input_tensor = self.donut_model.encoder.prepare_input(image_path, random_padding=self.split == "train")

        # input_ids
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        # processed_parse = self.task_prefix + " " + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.donut_model.decoder.tokenizer.eos_token 
        processed_parse = self.task_prefix + " " + "<extract_data_table>" + " " + self.prompt_end_token + " " + sample['table'] + self.donut_model.decoder.tokenizer.eos_token 
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse




class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    task_start_token
                    + self.donut_model.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + self.donut_model.decoder.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )


class DonutOri1MathQADataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.prompt_extra = "The provided function list is: ['numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum', 'numpy.add', 'numpy.subtract','numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=', 'numpy.==', 'numpy.!=','numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt', 'numpy.square', 'numpy.log','numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any', 'numpy.diff', 'numpy.corrcoef','numpy.cov','max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', 'getitem', 'select', '<', '<=', '>', '>=', '==', '!=']. You need to output in this format: <s_step1><s_func1>the function name you choose, must be in the provided function list</s_func1><s_arg1>the parameter list of the func1</s_arg1><s_output1>the variable name output by the func1</s_output1></s_step1><s_step2><s_func2>the function name you choose, must be in the provided function list</s_func2><s_arg2>the parameter list of the func2</s_arg2><s_output2>the variable name output by the func2</s_output2></s_step2>... Question: "
        
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = len(self.dataset_mathqa)
       
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.gt_token_sequences = []
        self.gt_token_sequences_idx_forimg = []

        for i in range(self.dataset_length):
            try:
                if i % 10 ==0:
                    print(i,', json2tkoen')
                sample = self.dataset_mathqa[i]
                gt_json = {"question":sample['query'],"answer":sample['ground_truth']}
                self.gt_token_sequences.append(
                        
                            task_start_token
                            + self.donut_model.json2token(
                                gt_json,
                                update_special_tokens_for_json_key=self.split == "train",
                                sort_json_key=self.sort_json_key,
                            )
                            + self.donut_model.decoder.tokenizer.eos_token
                        
                    )
                self.gt_token_sequences_idx_forimg.append(i)
            except:
                print('json2token error')
                continue

        

    def __len__(self) -> int:
        return self.dataset_length
        # return len(self.gt_token_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        
        data = self.dataset_mathqa
        
        index = self.gt_token_sequences_idx_forimg[index]
        sample = data[index]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")
        processed_parse = self.gt_token_sequences[index]
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse



class DonutOri2MathQADataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.prompt_extra = "The provided function list is: ['numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum', 'numpy.add', 'numpy.subtract','numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=', 'numpy.==', 'numpy.!=','numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt', 'numpy.square', 'numpy.log','numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any', 'numpy.diff', 'numpy.corrcoef','numpy.cov','max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', 'getitem', 'select', '<', '<=', '>', '>=', '==', '!=']. You need to output in this format: <s_step1><s_func1>the function name you choose, must be in the provided function list</s_func1><s_arg1>the parameter list of the func1</s_arg1><s_output1>the variable name output by the func1</s_output1></s_step1><s_step2><s_func2>the function name you choose, must be in the provided function list</s_func2><s_arg2>the parameter list of the func2</s_arg2><s_output2>the variable name output by the func2</s_output2></s_step2>... Question: "
        self.all_keys = ['question', 'answer', 'step3', 'arg1', 'arg8', 'output4', 'output5', 'step5', 'step2', 'func4', 'func1', 'output8', 'func5', 'func7', 'arg7', 'arg2', 'func3', 'step7', 'answer', 'step8', 'output3', 'step4', 'arg6', 'func2', 'func6', 'func8', 'arg4', 'arg3', 'arg5', 'output7', 'step1', 'output1', 'output2', 'step6', 'output6']
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = len(self.dataset_mathqa)
       
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.gt_token_sequences = []
        self.gt_token_sequences_idx_forimg = []

        for key in self.all_keys:
            tmp1 = "<s_" + key + ">"
            tmp2 = "</s_" + key + ">"
            self.donut_model.decoder.add_special_tokens([tmp1, tmp2])

        

        

    def __len__(self) -> int:
        return self.dataset_length
        # return len(self.gt_token_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        
        data = self.dataset_mathqa
        
        # idx = self.gt_token_sequences_idx_forimg[index]
        sample = data[index % len(data)]
        ground_truth = sample['ground_truth']
        # sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")

        # input_ids
        # processed_parse = self.gt_token_sequences[index]
        processed_parse = self.task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
                    
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse






class DonutOriMathQADataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        self.prompt_extra = "The provided function list is: ['numpy.argmax', 'numpy.argmin', 'numpy.max', 'numpy.min', 'numpy.sum', 'numpy.add', 'numpy.subtract','numpy.multiply', 'numpy.divide', 'numpy.<', 'numpy.<=', 'numpy.>', 'numpy.>=', 'numpy.==', 'numpy.!=','numpy.mean', 'numpy.median', 'numpy.std', 'numpy.var', 'numpy.abs', 'numpy.sqrt', 'numpy.square', 'numpy.log','numpy.exp', 'numpy.power', 'numpy.sort', 'numpy.delete', 'numpy.all', 'numpy.any', 'numpy.diff', 'numpy.corrcoef','numpy.cov','max', 'min', 'sum', 'len', 'str', 'int', 'float', 'abs', 'round', 'getitem', 'select', '<', '<=', '>', '>=', '==', '!=']. You need to output in this format: <s_step1><s_func1>the function name you choose, must be in the provided function list</s_func1><s_arg1>the parameter list of the func1</s_arg1><s_output1>the variable name output by the func1</s_output1></s_step1><s_step2><s_func2>the function name you choose, must be in the provided function list</s_func2><s_arg2>the parameter list of the func2</s_arg2><s_output2>the variable name output by the func2</s_output2></s_step2>... Question: "
        
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = len(self.dataset_mathqa)
       
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.gt_token_sequences = []
        self.gt_token_sequences_idx_forimg = []


    def __len__(self) -> int:
        return self.dataset_length
        # return len(self.gt_token_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        
        data = self.dataset_mathqa
        
        # idx = self.gt_token_sequences_idx_forimg[index]
        sample = data[index % len(data)]
        ground_truth = sample['ground_truth']
        # sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")

        # input_ids
        # processed_parse = self.gt_token_sequences[index]
        processed_parse = self.task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
                    
        # processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse



class DonutMultiTask2Dataset(Dataset):
    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        # task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.task_start_token = ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
        self.all_keys_mathqa1 = ['question', 'answer', 'step3', 'arg1', 'arg8', 'output4', 'output5', 'step5', 'step2', 'func4', 'func1', 'output8', 'func5', 'func7', 'arg7', 'arg2', 'func3', 'step7', 'answer', 'step8', 'output3', 'step4', 'arg6', 'func2', 'func6', 'func8', 'arg4', 'arg3', 'arg5', 'output7', 'step1', 'output1', 'output2', 'step6', 'output6']
        self.all_keys_mathqa2 = ['question', 'answer', 'step1', 'func1', 'arg1', 'output1', 'step2', 'func2', 'arg2', 'output2', 'step3', 'func3', 'arg3', 'output3', 'step4', 'func4', 'arg4', 'output4', 'step5', 'func5', 'arg5', 'output5', 'step6', 'func6', 'arg6', 'output6', 'step7', 'func7', 'arg7', 'output7', 'step8', 'func8', 'arg8', 'output8','step9', 'func9', 'arg9', 'output9','step10', 'func10', 'arg10', 'output10','step11', 'func11', 'arg11', 'output11','step12', 'func12', 'arg12', 'output12','step13', 'func13', 'arg13', 'output13','step14', 'func14', 'arg14', 'output14']
        self.all_keys_referringqa = ['arg2', 'step6', 'func6', 'step1', 'func4', 'step8', 'step4', 'output4', 'arg5', 'step5', 'step7', 'func2', 'arg4', 'output8', 'arg6', 'step2', 'func5', 'func7', 'answer', 'arg7', 'arg3', 'arg1', 'output6', 'output2', 'arg8', 'func8', 'func1', 'output7', 'output5', 'func3', 'step3', 'output1', 'output3']
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset_ocr = Img2TableDataset(
            split=self.split
        )
        self.dataset_summ = ChartSummDataset(
            split=self.split
        )
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        # self.dataset_mathqa1 = ChartMathQADataset1(
        #     split=self.split
        # )
        self.dataset_opencqa = ChartOpenQADataset(
            split=self.split
        )
        self.dataset_referringqa = ChartReferringQADataset(
            split=self.split
        )
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = max(len(self.dataset_ocr),len(self.dataset_summ),len(self.dataset_mathqa),len(self.dataset_opencqa),len(self.dataset_referringqa))
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        all_keys = self.all_keys_mathqa1
        for key in all_keys:
            tmp1 = "<s_" + key + ">"
            tmp2 = "</s_" + key + ">"
            self.donut_model.decoder.add_special_tokens([tmp1, tmp2])
        # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        for task_start_token in self.task_start_token:
            # ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
            self.donut_model.decoder.add_special_tokens([task_start_token, self.prompt_end_token])
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        choices = ['ocr','summ','mathqa','opencqa','referringqa']
        # choices = [self.dataset_ocr, self.dataset_summ, self.dataset_mathqa, self.dataset_opencqa]
        weights = [1, 2, 5, 3, 5]
        choose = random.choices(choices, weights=weights)[0]
        if 'ocr' in choose:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        elif 'summ' in choose:
            task_start_token = self.task_start_token[1]
            data = self.dataset_summ
        elif 'opencqa' in choose:
            task_start_token = self.task_start_token[2]
            data = self.dataset_opencqa
        elif 'mathqa' in choose:
            task_start_token = self.task_start_token[3]
            data = self.dataset_mathqa
        # elif 'mathqa1' in choose:
        #     task_start_token = self.task_start_token[3]
        #     data = self.dataset_mathqa1
        elif 'referringqa' in choose:
            task_start_token = self.task_start_token[4]
            data = self.dataset_referringqa
        else:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        # data = self.dataset_summ
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']

        if 'mathqa' in choose or 'referringqa' in choose:
            ground_truth = sample['ground_truth']
        # input_tensor
        else:
            # <s_question>Across all years, what is the maximum trade with high income economies(%) of the yellow arrows?</s_question><s_answer></s_answer>
            ground_truth = "<s_question>" + str(sample['query']) + "</s_question>" + "<s_answer>" + str(sample['ground_truth']) + "</s_answer>"
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        processed_parse = task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class DonutMultiTask2NoCOTDataset(Dataset):
    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        # task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.task_start_token = ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
        self.all_keys_mathqa1 = ['question', 'answer', 'step3', 'arg1', 'arg8', 'output4', 'output5', 'step5', 'step2', 'func4', 'func1', 'output8', 'func5', 'func7', 'arg7', 'arg2', 'func3', 'step7', 'answer', 'step8', 'output3', 'step4', 'arg6', 'func2', 'func6', 'func8', 'arg4', 'arg3', 'arg5', 'output7', 'step1', 'output1', 'output2', 'step6', 'output6']
        self.all_keys_mathqa2 = ['question', 'answer', 'step1', 'func1', 'arg1', 'output1', 'step2', 'func2', 'arg2', 'output2', 'step3', 'func3', 'arg3', 'output3', 'step4', 'func4', 'arg4', 'output4', 'step5', 'func5', 'arg5', 'output5', 'step6', 'func6', 'arg6', 'output6', 'step7', 'func7', 'arg7', 'output7', 'step8', 'func8', 'arg8', 'output8','step9', 'func9', 'arg9', 'output9','step10', 'func10', 'arg10', 'output10','step11', 'func11', 'arg11', 'output11','step12', 'func12', 'arg12', 'output12','step13', 'func13', 'arg13', 'output13','step14', 'func14', 'arg14', 'output14']
        self.all_keys_referringqa = ['arg2', 'step6', 'func6', 'step1', 'func4', 'step8', 'step4', 'output4', 'arg5', 'step5', 'step7', 'func2', 'arg4', 'output8', 'arg6', 'step2', 'func5', 'func7', 'answer', 'arg7', 'arg3', 'arg1', 'output6', 'output2', 'arg8', 'func8', 'func1', 'output7', 'output5', 'func3', 'step3', 'output1', 'output3']
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset_ocr = Img2TableDataset(
            split=self.split
        )
        self.dataset_summ = ChartSummDataset(
            split=self.split
        )
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        # self.dataset_mathqa1 = ChartMathQADataset1(
        #     split=self.split
        # )
        self.dataset_opencqa = ChartOpenQADataset(
            split=self.split
        )
        self.dataset_referringqa = ChartReferringQADataset(
            split=self.split
        )
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = max(len(self.dataset_ocr),len(self.dataset_summ),len(self.dataset_mathqa),len(self.dataset_opencqa),len(self.dataset_referringqa))
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        all_keys = self.all_keys_mathqa1
        for key in all_keys:
            tmp1 = "<s_" + key + ">"
            tmp2 = "</s_" + key + ">"
            self.donut_model.decoder.add_special_tokens([tmp1, tmp2])
        # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        for task_start_token in self.task_start_token:
            # ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
            self.donut_model.decoder.add_special_tokens([task_start_token, self.prompt_end_token])
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        choices = ['ocr','summ','mathqa','opencqa','referringqa']
        # choices = [self.dataset_ocr, self.dataset_summ, self.dataset_mathqa, self.dataset_opencqa]
        weights = [1, 2, 5, 3, 5]
        choose = random.choices(choices, weights=weights)[0]
        if 'ocr' in choose:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        elif 'summ' in choose:
            task_start_token = self.task_start_token[1]
            data = self.dataset_summ
        elif 'opencqa' in choose:
            task_start_token = self.task_start_token[2]
            data = self.dataset_opencqa
        elif 'mathqa' in choose:
            task_start_token = self.task_start_token[3]
            data = self.dataset_mathqa
        # elif 'mathqa1' in choose:
        #     task_start_token = self.task_start_token[3]
        #     data = self.dataset_mathqa1
        elif 'referringqa' in choose:
            task_start_token = self.task_start_token[4]
            data = self.dataset_referringqa
        else:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        # data = self.dataset_summ
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']

        if 'mathqa' in choose or 'referringqa' in choose:
            ground_truth = sample['ground_truth']
        # input_tensor
        else:
            # <s_question>Across all years, what is the maximum trade with high income economies(%) of the yellow arrows?</s_question><s_answer></s_answer>
            ground_truth = "<s_question>" + str(sample['query']) + "</s_question>" + "<s_answer>" + str(sample['ground_truth']) + "</s_answer>"
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        processed_parse = task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse
    

class DonutMultiTask2NoReferringDataset(Dataset):
    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        # task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.task_start_token = ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>"]
        self.all_keys_mathqa1 = ['question', 'answer', 'step3', 'arg1', 'arg8', 'output4', 'output5', 'step5', 'step2', 'func4', 'func1', 'output8', 'func5', 'func7', 'arg7', 'arg2', 'func3', 'step7', 'answer', 'step8', 'output3', 'step4', 'arg6', 'func2', 'func6', 'func8', 'arg4', 'arg3', 'arg5', 'output7', 'step1', 'output1', 'output2', 'step6', 'output6']
        self.all_keys_mathqa2 = ['question', 'answer', 'step1', 'func1', 'arg1', 'output1', 'step2', 'func2', 'arg2', 'output2', 'step3', 'func3', 'arg3', 'output3', 'step4', 'func4', 'arg4', 'output4', 'step5', 'func5', 'arg5', 'output5', 'step6', 'func6', 'arg6', 'output6', 'step7', 'func7', 'arg7', 'output7', 'step8', 'func8', 'arg8', 'output8','step9', 'func9', 'arg9', 'output9','step10', 'func10', 'arg10', 'output10','step11', 'func11', 'arg11', 'output11','step12', 'func12', 'arg12', 'output12','step13', 'func13', 'arg13', 'output13','step14', 'func14', 'arg14', 'output14']
        self.all_keys_referringqa = ['arg2', 'step6', 'func6', 'step1', 'func4', 'step8', 'step4', 'output4', 'arg5', 'step5', 'step7', 'func2', 'arg4', 'output8', 'arg6', 'step2', 'func5', 'func7', 'answer', 'arg7', 'arg3', 'arg1', 'output6', 'output2', 'arg8', 'func8', 'func1', 'output7', 'output5', 'func3', 'step3', 'output1', 'output3']
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset_ocr = Img2TableDataset(
            split=self.split
        )
        self.dataset_summ = ChartSummDataset(
            split=self.split
        )
        self.dataset_mathqa = ChartMathQADataset(
            split=self.split
        )
        # self.dataset_mathqa1 = ChartMathQADataset1(
        #     split=self.split
        # )
        self.dataset_opencqa = ChartOpenQADataset(
            split=self.split
        )
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = max(len(self.dataset_ocr),len(self.dataset_summ),len(self.dataset_mathqa),len(self.dataset_opencqa))
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        all_keys = self.all_keys_mathqa2
        for key in all_keys:
            tmp1 = "<s_" + key + ">"
            tmp2 = "</s_" + key + ">"
            self.donut_model.decoder.add_special_tokens([tmp1, tmp2])
        # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        for task_start_token in self.task_start_token:
            # ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
            self.donut_model.decoder.add_special_tokens([task_start_token, self.prompt_end_token])
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        choices = ['ocr','summ','mathqa','opencqa']
        # choices = [self.dataset_ocr, self.dataset_summ, self.dataset_mathqa, self.dataset_opencqa]
        weights = [1, 2, 5, 3]
        choose = random.choices(choices, weights=weights)[0]
        if 'ocr' in choose:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        elif 'summ' in choose:
            task_start_token = self.task_start_token[1]
            data = self.dataset_summ
        elif 'opencqa' in choose:
            task_start_token = self.task_start_token[2]
            data = self.dataset_opencqa
        elif 'mathqa' in choose:
            task_start_token = self.task_start_token[3]
            data = self.dataset_mathqa
        else:
            task_start_token = self.task_start_token[0]
            data = self.dataset_ocr
        # data = self.dataset_summ
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']

        if 'mathqa' in choose:
            ground_truth = sample['ground_truth']
        # input_tensor
        else:
            # <s_question>Across all years, what is the maximum trade with high income economies(%) of the yellow arrows?</s_question><s_answer></s_answer>
            ground_truth = "<s_question>" + str(sample['query']) + "</s_question>" + "<s_answer>" + str(sample['ground_truth']) + "</s_answer>"
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        processed_parse = task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse

        
class DonutOCRDataset(Dataset):
    def __init__(
        self,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        # task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.task_start_token = ["<s_ocr>"]
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.dataset_ocr = Img2TableDataset(
            split=self.split
        )
        
        #list(dict)
        # self.dataset_length = len(self.dataset_ocr) + len(self.dataset_summ) + len(self.dataset_mathqa) + len(self.dataset_opencqa)
        self.dataset_length = max(len(self.dataset_ocr))
        # self.task_prefix = task_prefix
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        for task_start_token in self.task_start_token:
            # ["<s_ocr>","<s_summ>","<s_opencqa>","<s_mathqa>","<s_referringqa>"]
            self.donut_model.decoder.add_special_tokens([task_start_token, self.prompt_end_token])
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        task_start_token = self.task_start_token[0]
        data = self.dataset_ocr
        
        sample = data[index % len(data)]
        image = sample['image']
        task_prefix = sample['task']

        ground_truth = "<s_question>" + str(sample['query']) + "</s_question>" + "<s_answer>" + str(sample['ground_truth']) + "</s_answer>"
        
        input_tensor = self.donut_model.encoder.prepare_input(image, random_padding=self.split == "train")

        processed_parse = task_start_token + ground_truth + self.donut_model.decoder.tokenizer.eos_token
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse