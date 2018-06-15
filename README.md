# CIFAR-10-PYTORCH
## I achieved **83%** on the test set, in the following code you can choose between transfer learning model (using Resnet18) and my own model which is :

1)Conv layer -> 3 channels to 50 with kernel size of 5
2)BatchNorm2d
3)Activation function (Relu)
4)Conv layer -> 50 channels to 150 with kernel size of 5
5)BatchNorm2d
6)Activation function (Relu)
7)Linear
8)BatchNorm1d
9)Activation(Relu)
10)Dropout p = 0.5
11)Linear
12)BatchNorm1d
13)Activation(Relu)
14)Dropout p = 0.25
15)Linear
16)LogSoftmax

