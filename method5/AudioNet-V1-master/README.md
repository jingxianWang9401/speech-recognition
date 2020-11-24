# AudioNet-V1
AudioNet is a simple convolutional neural net based on 1-D convolutions. This is trained and tested on google's speech command dataset.
AudioNet是一种基于一维卷积的简单卷积神经网络，这是在谷歌的语音命令数据集上训练和测试的。


## Requirements

Tested with following setup

### Software
1) Python 3.5
2) Numpy
3) Scipy
4) Keras 2.0.8
5) Tensorflow 1.4.1
6) Scikit-learn

### Hardware

1) GTX 1050 TI 4 GB

## One Dimensional CNN
Here, 1-D convolutions (linear convolutions) are used on top of regular hidden layers to classify the speech signal. The dataset used is Google's speech Commands Dataset
一维CNN             
在这里，一维卷积（线性卷积）被用于在规则的隐藏层之上对语音信号进行分类。使用的数据集是Google的speech Commands数据集

The network has five 1-D convolutional layers with kernel size 32 and stride of 4. They are followed by four hidden layers with 512 neurons each. The network has approximately 10 million parameters in total.

该网络有5个一维卷积层，核大小32，步长4。接着是四个隐藏层，每个层有512个神经元。该网络总共有大约1000万个参数。


### Data Augmentation used

1) Random noise
2) Random shift

###使用的数据扩充              
1） 随机噪声             
2） 随机移位


### Training Loss vs Epochs

![](https://github.com/vj-1988/AudioNet-V1/blob/master/Images/training_loss.png)

### Training Acuracy vs Epochs

![](https://github.com/vj-1988/AudioNet-V1/blob/master/Images/training_accuracy.png)


## Training 

数据集必须位于适当的子文件夹中，每个文件夹名称都是类标签。脚本AudioNet32.py需要以下输入来训练：
The dataset has to be in appropriate subfolders with each folder name being the class label. The script AudioNet32.py needs the following inputs to train

1) data_path : root folder of dataset
2) train_ratio : ratio of files to be used for training and remaining is for validation
3) batch_size : minibatch size for training.
4) num_epochs : total no. of epochs
5) dst : destination folder to save weights, logs

The script will generate a pickle file that contains synset for validation, training and validation files path and labels. This can be used to resume training using resume_training() function.
该脚本将生成一个pickle文件，其中包含验证、培训和验证文件路径和标签的synset。这可用于使用resume_training（）函数恢复训练。

The script will save weights once in every 2 epochs.

## Validation

The synset used for training is available in train_data_dic.pkl file. The pretrained weights are available in the following link
用于训练的synset在train_data_dic.pkl文件中提供。以下链接提供了预训练的重量

[Download pretrained weights (Epoch 10)](https://drive.google.com/file/d/1vrfeXhtb8mRiLI1ja3Ep80M-zv9a-IVw/view?usp=sharing)

