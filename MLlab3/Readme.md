# Readme

这个文件夹包括

> lab3.py：使用的是Resnet101进行预训练模型
>
> lab3_google.py：使用的是GoogLeNet进行预训练模型
>
> lab3_cnn.py：使用的是我自己实现的CNN来训练
>
> prediction.py：是进行测试模型的脚本
>
> MLlab3.md：实验报告，包括了调参实验和模型训练的全过程
>
> 以及一些markdown文本所需的图片

训练集在该目录的lab3_data/data中，可以看上面三个训练模型的脚本看出

prediction.py需要根据测试对应的模型修改 `ClassificationModel`类和 `VideoDataset`类，并使用自己使用的视频路径进行测试。

三个训练模型文件下载好依赖之后，直接 `python  .py`运行即可。

生成的模型位于该目录下，名称分别为 `Resnet101pretrained_model.pth`，`Googlepretrained_model.pth`，`cnn_model.pth`

同时，每次运行模型由于执行了 `plt.save`，因此会生成对应的loss曲线以及精确度曲线。
