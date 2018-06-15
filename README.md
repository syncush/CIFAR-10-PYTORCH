# CIFAR-10-PYTORCH
## I achieved **83%** on the test set, in the following code you can choose between transfer learning model (using Resnet18) and my own model which is :

1)Conv layer -> 3 channels to 50 with kernel size of 5<br/>
2)BatchNorm2d<br/>
3)Activation function (Relu)<br/>
4)Conv layer -> 50 channels to 150 with kernel size of 5<br/>
5)BatchNorm2d<br/>
6)Activation function (Relu)<br/>
7)Linear<br/>
8)BatchNorm1d<br/>
9)Activation(Relu)<br/>
10)Dropout p = 0.5<br/>
11)Linear<br/>
12)BatchNorm1d<br/>
13)Activation(Relu)<br/>
14)Dropout p = 0.25<br/>
15)Linear<br/>
16)LogSoftmax<br/>

