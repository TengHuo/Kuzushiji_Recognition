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
                                     kernel_size=7).to(device)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=128,
                                     kernel_size=6).to(device)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(in_features=128*8*8,
                                  out_features=4212,
                                  bias=True).to(device)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1).to(device)

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
classify_model.load_state_dict('../cache/classify_model.pth', map_location=device)
classify_model.eval()
scale_resize = (48, 48)

uc_translation = pd.read_csv("../cache/my_unicode_translation.csv")
uc_list = uc_translation["Unicode"].values.tolist()


def classify(img_array, boxes):
    img = Image.fromarray(img_array)
    inputs = []
    for x, y, w, h in boxes:
        # 这里的坐标应该是原始图片上的坐标（？？？？）
        # Crop as Character's PIL Image
        char_img = img.crop((x, y, x + w, y + h))
        # Resize Character's PIL Image
        char_img = char_img.resize(scale_resize)
        # Gray-Scale Character's PIL Image where the channel is 1
        char_img = char_img.convert('L')
        # Convert from Character's PIL Image to Tensor
        char_img = torchvision.transforms.functional.to_tensor(char_img)
        inputs.append(char_img)
    inputs = torch.stack(inputs)
    outputs = classify_model(inputs)
    outputs = torch.argmax(outputs, 1)

    chars = []
    for idx in outputs:
        chars.append(uc_list[idx])
    return chars


if __name__ == '__main__':
    # 测试
    #
    pass
