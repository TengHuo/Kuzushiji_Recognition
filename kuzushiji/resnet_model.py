# -*- coding: utf-8 -*-
# resnet_model.py
# @Time     : 13/Oct/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#
from torch import nn


class CharacterResnet(nn.Module):
    def __init__(self, model):
        super(CharacterResnet, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_ = nn.Linear(2048, 4212)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)

        return x
