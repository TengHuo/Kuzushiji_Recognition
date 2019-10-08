# Kuzushiji_Recognition

项目地址：https://www.kaggle.com/c/kuzushiji-recognition

## 目录结构

cache和input文件夹需要手动建立，从kaggle下载的数据放到input中，cache文件夹会保存一些模型数据和中间结果。

notebooks里是主要的代码，kuzushiji文件夹中封装了一些在notebooks中调用的函数。

```bash
.
├── README.md
├── cache
├── input
│   ├── kuzushiji-recognition.zip
│   ├── sample_submission.csv
│   ├── test_images
│   ├── train.csv
│   ├── train_images
│   └── unicode_translation.csv
├── kuzushiji
│   ├── __init__.py
│   ├── classify_character.py
│   └── utils
└── notebooks
    ├── NotoSansCJKjp-Regular.otf
    ├── pytorch_classification_demo.ipynb
    ├── test-resnet.ipynb
    ├── test.ipynb
    ├── unet-character-detector.ipynb
    └── visualisation.ipynb
```

## Requirements


Python 3.6.9
Numpy
Pandas
Matplotlib
PIL
PyTorch
Torchvision
tqdm
scikit-image
scikit-learn
Jupyter

## Getting Started


## Reference

1. https://www.kaggle.com/christianwallenwein/visualization-useful-functions-small-eda
2. https://www.kaggle.com/infhyroyage/create-pytorch-dataset-for-classifying-characters
3. https://www.kaggle.com/kmat2019/centernet-keypoint-detector

其他

1. https://zhuanlan.zhihu.com/p/64295374
2. https://github.com/xingyizhou/CenterNet
3. https://www.infoq.cn/article/XUDiNPviWhHhvr6x_oMv
4. https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb



