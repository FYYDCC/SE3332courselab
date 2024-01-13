# Readme

#### 一、实验环境

> PyTorch 2.0.0
>
> Python 3.8(ubuntu20.04)
>
> Cuda 11.8

显卡：RTX 4090(24GB) * 1

#### 二、代码运行

提交的代码全部保存在code文件夹中

requirement.txt

>包含了需要额外安装的安装包
>
>```
>pip install -r requirements.txt
>```

data.py

>主要用于定义和处理数据集，包括数据的加载、预处理以及数据集类的定义。在这里，它用于加载源代码数据集，进行 tokenization，并创建一个用于模型训练的 PyTorch 数据集类。

model.py

> `model.py` 文件用于定义深度学习模型的架构。它定义了一个 `DistilGPT-2`模型，包括模型的层次结构、前向传播逻辑等。

trainer.py

> `trainer.py` 文件通常包含一个模型训练的类，这个类封装了一些用于训练模型的功能，例如训练一个 epoch、计算损失、执行反向传播等。它提供了一个更高层次的接口，用于将模型、数据加载器等组合在一起并执行训练。

train.py

>`train.py` 文件是主要的训练脚本，负责组织和执行模型的训练过程。它包括训练循环、优化器的设置、损失计算等。通过运行这个脚本，你可以启动训练过程。

运行方法：（确保已经安装transformers模型）

这里默认已经自己构造好了数据集并下载了distilGpt2到同一文件夹路径。

要开始训练模型，需要在命令行中运行

```
python train.py --model_select distilgpt2
```

训练完毕之后，需要使用训练好的模型，在命令行中执行

```
python interact.py
```

