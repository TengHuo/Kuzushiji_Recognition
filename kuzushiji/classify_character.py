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


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=16,
                                     kernel_size=7)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=128,
                                     kernel_size=6)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(in_features=128*8*8,
                                  out_features=4212,
                                  bias=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x) # (batch, 1, 48, 48) -> (batch, 16, 42, 42)
        out = self.relu1(out)
        out = self.maxpool1(out) # (batch, 16, 42, 42) -> (batch, 16, 21, 21)
        out = self.conv2(out) # (batch, 16, 21, 21) -> (batch, 128, 16, 16)
        out = self.relu2(out)
        out = self.maxpool2(out) # (batch, 128, 16, 16) -> (batch, 128, 8, 8)
        out = out.view(out.size(0), -1) # (batch, 128, 8, 8) -> (batch, 8192)
        out = self.fc(out) # (batch, 8192) -> (batch, 4212)
        out = self.log_softmax(out)
        return out


classify_model = DemoModel()
device = torch.device("cpu")
classify_model.load_state_dict(torch.load('../cache/classify_model.pth', map_location=device))
classify_model.eval()
scale_resize = (48, 48)

uc_translation = pd.read_csv("../cache/my_unicode_translation.csv")
uc_list = uc_translation["Unicode"].values.tolist()


def classify(image: Image, rects: list):
    images = []
    for x, y, w, h in rects:
        # 这里的坐标应该是原始图片上的坐标（？？？？）
        # Crop as Character's PIL Image
        char_img = image.crop((x, y, x + w, y + h))
        # Resize Character's PIL Image
        char_img = char_img.resize(scale_resize)
        # Gray-Scale Character's PIL Image where the channel is 1
        char_img = char_img.convert('L')
        # Convert from Character's PIL Image to Tensor
        char_img = torchvision.transforms.functional.to_tensor(char_img)
        images.append(char_img)
    images = torch.stack(images)
    outputs = classify_model(images)
    outputs = torch.argmax(outputs, 1)

    chars = []
    for idx in outputs:
        chars.append(uc_list[idx])
    return chars


__all__ = ["classify"]


if __name__ == '__main__':
    train_df = pd.read_csv('../input/train.csv')

    img_id = "100241706_00009_1"
    xy = train_df[train_df['image_id'] == img_id]
    labels = xy['labels'].iloc[0].split(" ")

    test_boxes = []
    test_y = []
    for i in range(0, len(labels), 5):
        uc = labels[i]
        test_y.append(uc)
        x = int(labels[i + 1])
        y = int(labels[i + 2])
        w = int(labels[i + 3])
        h = int(labels[i + 4])
        test_boxes.append((x, y, w, h))

    test_img = Image.open('../input/train_images/{}.jpg'.format(img_id))
    result = classify(test_img, test_boxes)
    correct = 0
    for i in range(len(result)):
        if result[i] == test_y[i]:
            correct += 1
    print(correct/len(result))

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
