# -*- coding: utf-8 -*-
# classify_character.py
# @Time     : 05/Oct/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import random
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision
from PIL import Image
import torchvision.models as models
from albumentations import (Compose, ShiftScaleRotate, RGBShift, Cutout, RandomCrop, PadIfNeeded, Resize)
import cv2
from .resnet_model import CharacterResnet


h_resize = 64
w_resize = 64


def data_augment():
    return Compose([
        PadIfNeeded(min_height=90, min_width=70, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        Resize(height=h_resize, width=w_resize, always_apply=True)
    ], p=1)


image_transform = data_augment()
resnet = models.resnet50(pretrained=True)
classify_model = CharacterResnet(resnet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classify_model.load_state_dict(torch.load('../checkpoints/resnet50_3.pth'))
classify_model.to(device)
classify_model.eval()
scale_resize = (h_resize, w_resize)

uc_translation = pd.read_csv("../cache/my_unicode_translation.csv")
uc_list = uc_translation["Unicode"].values.tolist()


def classify(image: Image, rects: list):
    images = []
    centers = []
    boxes = []
    for x, y, w, h in rects:
        # Crop as Character's PIL Image
        char_img = image.crop((x, y, x + w, y + h))
        # Resize Character's PIL Image
        char_img = np.array(char_img)
        if char_img.shape == ():
            continue
        char_img = image_transform(image=char_img)['image']
        char_img = torchvision.transforms.functional.to_tensor(char_img)
        images.append(char_img)
        centers.append([x + w/2, y + h/2])
        boxes.append([x, y, w, h])
    images = torch.stack(images)
    images = images.to(device)
    outputs = classify_model(images)
    outputs = outputs.cpu()
    outputs = torch.argmax(outputs, 1)
    chars = []
    for idx in outputs:
        chars.append(uc_list[idx])
    chars = np.array(chars).reshape((-1, 1))
    centers = np.array(centers).astype(np.int)
    centers = centers.astype(str)
    boxes = np.array(boxes).astype(np.int)
    boxes = boxes.astype(str)
    return chars, centers, boxes

    # center_labels = np.concatenate((chars, centers), axis=1)
    # center_labels = center_labels.flatten()
    # box_labels = np.concatenate((chars, boxes), axis=1)
    # box_labels = box_labels.flatten()
    # return " ".join(center_labels), " ".join(box_labels)


__all__ = ["classify"]


if __name__ == '__main__':
    train_df = pd.read_csv('../input/train.csv')

    img_id = "100241706_00009_1"
    xy = train_df[train_df['image_id'] == img_id]
    label_list = xy['labels'].iloc[0].split(" ")

    test_boxes = []
    test_y = []
    for i in range(0, len(label_list), 5):
        uc = label_list[i]
        test_y.append(uc)
        x = int(label_list[i + 1])
        y = int(label_list[i + 2])
        w = int(label_list[i + 3])
        h = int(label_list[i + 4])
        test_boxes.append((x, y, w, h))

    test_img = Image.open('../input/train_images/{}.jpg'.format(img_id))
    result_centers, result_boxes = classify(test_img, test_boxes)
    print(result_centers)
    print(result_boxes)
    # correct = 0
    # for i in range(len(result)):
    #     if result[i] == test_y[i]:
    #         correct += 1
    # print(correct/len(result))

    # avg_acc = []
    # for j in tqdm(range(0, 10)): #train_df['image_id'].shape[0])):
    #     xy = train_df.iloc[j]
    #     img_id = xy['image_id']
    #     labels = xy['labels']
    #     if labels is not np.nan:
    #         labels = labels.split(" ")
    #     else:
    #         continue
    #     test_boxes = []
    #     test_y = []
    #     for i in range(0, len(labels), 5):
    #         uc = labels[i]
    #         test_y.append(uc)
    #         x = int(labels[i+1])
    #         y = int(labels[i+2])
    #         w = int(labels[i+3])
    #         h = int(labels[i+4])
    #         test_boxes.append((x, y, w, h))
    #
    #     test_img = Image.open('../input/train_images/{}.jpg'.format(img_id))
    #     result = classify(test_img, test_boxes)
    #     correct = 0
    #     for i in range(len(result)):
    #         if result[i] == test_y[i]:
    #             correct += 1
    #     avg_acc.append(correct/len(result))
    # print(avg_acc)
    # print(np.average(avg_acc))
