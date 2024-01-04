"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from donut import Img2TableDataset
from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def test(args):
    # pretrained_model_name_or_path = "/mnt/petrelfs/donut/result/donut_base/binmodel_multitask3/epoch3"
    pretrained_model_name_or_path = "/mnt/petrelfs/donut/result/donut_base/binmodel_multitask3/epoch3"
    # pretrained_model_name_or_path = "/mnt/petrelfs/donut/result/donut_base/binmodel_multitask3_norefer1/epoch1"
    # pretrained_model_name_or_path = "/mnt/petrelfs/donut/result/donut_base/binmodel_multitask3_noocr2/epoch0"
    # pretrained_model_name_or_path = "/mnt/petrelfs/donut/result/donut_base/binmodel_nocot/epoch3"
    pretrained_model = DonutModel.from_pretrained(pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()


    predictions = []
    ground_truths = []
    result = []
    accs = []
    split = 'test'
    test_dataset = Img2TableDataset(
                    split=split,
                )
            

    # dataloader = torch.utils.data.DataLoader(
    #                 test_dataset,
    #                 batch_size=1,
    #                 num_workers = 8,
    #                 pin_memory=True,
    #                 shuffle=False,
    #             )
    for idx, sample in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        if sample is None:
            continue

        # pred = pretrained_model.inference(
        #     image_tensors=image_tensors,
        #     prompt_tensors=decoder_prompts,
        #     return_json=False,
        #     return_attentions=False,
        # )["predictions"]
        try:
            # pred = pretrained_model.inference(image=sample["image"], prompt=str(sample['query']) + "<s_answer>")["predictions"][0]
            pred = pretrained_model.inference(
                image=sample["image"],
                prompt=f"<s_ocr><s_question>" + str(sample['query']) + "</s_question><s_answer>",
            )["predictions"][0]
            if 'answer' in list(pred.keys()):
                pred = pred['answer']
                ground_truth = sample['ground_truth']
                imgname = sample['imgname']
                # answer_gt = sample['answer']
                print('pred: ',pred)
                print('gt: ',ground_truth)
                # print('answer_gt: ',answer_gt)
                print('\n')
                result.append({'pred':pred,'gt':ground_truth,'img':imgname })
        except:
            print(idx,ground_truth)
            print('error')
            continue


    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    # if args.task_name is None:
        # args.task_name = os.path.basename(args.dataset_name_or_path)

    result = test(args)

    # with open('/mnt/petrelfs/donut/test_mathqa_plotqa_beam4.json','w') as f:
    #     json.dump(result,f)
    with open('/mnt/petrelfs/donut/result/donut_base/binmodel_multitask3/test_plotqa_ocr.json','w') as f:
        json.dump(result,f)
