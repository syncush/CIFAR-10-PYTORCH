# CIFAR-10-PYTORCH
## I achieved **83%** on the test set, in the following code you can choose between transfer learning model (using Resnet18) and my own model which is :

1)Conv layer -> 3 channels to 50 with kernel size of 5<br/>
2)BatchNorm2d<br/>
3)Activation function (Relu)<br/>
4)Max Pooling -> Stride  = 2<br/>
5)Conv layer -> 50 channels to 150 with kernel size of 5<br/>
6)BatchNorm2d<br/>
7)Activation function (Relu)<br/>
8)Max Pooling -> Stride  = 2 <br/>
9)Linear 3500 to 512 <br/>
10)BatchNorm1d<br/>
11)Activation(Relu)<br/>
12)Dropout p = 0.5<br/>
13)Linear -> 512 to 256 <br/>
14)BatchNorm1d<br/>
15)Activation(Relu)<br/>
16)Dropout p = 0.25<br/>
17)Linear 256 to 10<br/>
18)LogSoftmax<br/>

