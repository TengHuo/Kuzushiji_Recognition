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


class DemoModel(torch.nn.Module):
    def __init__(self, model):
        super(DemoModel, self).__init__()
        self.resnet_layer = torch.nn.Sequential(*list(model.children())[:-2])
        self.fc_ = torch.nn.Linear(2048, 4212)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        return x


resnet = models.resnet18(pretrained=True)
classify_model = DemoModel(resnet)
device = torch.device("cpu")
classify_model.load_state_dict(torch.load('../cache/classify_resnet18.pth', map_location=device))
classify_model.eval()
scale_resize = (64, 64)

uc_translation = pd.read_csv("../cache/my_unicode_translation.csv")
uc_list = uc_translation["Unicode"].values.tolist()


def classify(image: Image, rects: list):
    images = []
    for x, y, w, h in rects:
        # Crop as Character's PIL Image
        char_img = image.crop((x, y, x + w, y + h))
        # Resize Character's PIL Image
        char_img = char_img.resize(scale_resize)
        # # Gray-Scale Character's PIL Image where the channel is 1
        # char_img = char_img.convert('L')
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
