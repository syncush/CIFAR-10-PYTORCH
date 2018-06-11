from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision.models import resnet18
import torch.nn.functional as F
import torch.cuda
import time
import datetime
import pickle
import sys
from torch.autograd.variable import Variable
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch import nn
import itertools
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_datasets_cifar10(resnet=False):
    if not resnet:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_dataset = CIFAR10(root='./data/cifar10_train',
                                train=True,
                                transform=transform,
                                download=True)

        test_dataset = CIFAR10(root='./data/cifar10_test',
                               train=False,
                               transform=transform_test,
                               download=True)
        return train_dataset, test_dataset
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_dataset = CIFAR10(root='./data/cifar10_train',
                                train=True,
                                transform=transform,
                                download=True)

        test_dataset = CIFAR10(root='./data/cifar10_test',
                               train=False,
                               transform=transform_test,
                               download=True)
        return train_dataset, test_dataset


def get_data_loaders_cifar10(batch_size, resnet=False):
    train_dataset, test_dataset = get_datasets_cifar10(resnet)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def confusion_matrix_creator(model, loader):
    preds = []
    real = []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        preds.append(str(pred.item()))
        real.append(str(target.item()))
    cm = confusion_matrix(real, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(list(range(0, 10))))
    plt.xticks(tick_marks, list(range(0, 10)), rotation=45)
    plt.yticks(tick_marks, list(range(0, 10)))

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


class ConvulutionalNeuralNetwork(nn.Module):
    def __init__(self, channel_output_one=20, channel_output_two=40, layer_one_out=500, layer_two_out=50):
        super(ConvulutionalNeuralNetwork, self).__init__()
        kernel_size = 5
        self.size_channel_out = channel_output_two * 5 * 5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channel_output_one, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=channel_output_one, out_channels=channel_output_two, kernel_size=kernel_size)
        self.batchnorm2d1 = nn.BatchNorm2d(channel_output_one)
        self.batchnorm2d2 = nn.BatchNorm2d(channel_output_two)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation4 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(layer_one_out)
        self.bn2 = nn.BatchNorm1d(layer_two_out)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(self.size_channel_out, layer_one_out)
        self.linear2 = nn.Linear(layer_one_out, layer_two_out)
        self.linear3 = nn.Linear(layer_two_out, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # First layer
        out = self.conv1(x)
        out = self.batchnorm2d1(out)
        out = self.activation1(out)
        out = self.pooling1(out)

        # Second layer
        out = self.conv2(out)
        out = self.batchnorm2d2(out)
        out = self.activation2(out)
        out = self.pooling2(out)
        ###
        out = out.view(-1, self.size_channel_out)
        ###
        # third layer
        out = self.linear1(out)
        out = self.bn1(out)
        out = self.activation3(out)
        out = self.dropout1(out)

        # fourth layer
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.activation4(out)
        out = self.dropout2(out)

        # Output layer
        out = self.linear3(out)
        return self.softmax(out)


def tavi_resnet(batch_size):
    model_conv = resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    model_conv.fc = nn.Linear(model_conv.fc.in_features, 10)
    return get_data_loaders_cifar10(batch_size, True), model_conv


class Trainer(object):
    def train_one_epoc(self):
        raise NotImplemented

    def test(self, to_test_by):
        raise NotImplemented

    def train(self, dictionary):
        raise NotImplemented

    def write_test_pred(self, accuracy):
        raise NotImplemented


class TrainerModels(object):
    def __init__(self, model, num_epocs, loaders, crit, optimizer, model_no, batch_size):
        super(TrainerModels, self).__init__()
        self.model = model
        self.num_epocs = num_epocs
        self.train_loader, self.test_loader = loaders
        self.criterion = crit
        self.optimizer = optimizer
        self.model_no = model_no
        self.batch_size = batch_size

    def train_one_epoc(self):
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(device), labels.to(device)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = self.model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = self.criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            self.optimizer.step()

    def test(self, to_test_by):
        to_test_by_loader, to_test_by_name = to_test_by
        self.model.eval()
        test_loss = 0.0
        correct = 0.0
        for data, target in to_test_by_loader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()
        len_data = len(to_test_by_loader) if to_test_by_name is not 'train' else len(
            to_test_by_loader) * self.batch_size
        test_loss /= len_data
        print("Info: correct {} / {}".format(correct, len_data))
        return test_loss, float(100.0 * correct) / len_data

    def train(self, dictionary):
        test_loss_list, test_acc_list = [], []
        train_acc_list, train_loss_list = [], []
        test_loss, test_acc = 0.0, 0.0
        for x in range(1, self.num_epocs + 1):
            print("Epoc # {}".format(x))
            if x % 5 is 0 and x is not 0 and x < 20:
                self.optimizer = SGD(self.model.parameters(), lr=0.1 ** int((x/5)), momentum=0.9, weight_decay=5e-4)
            self.train_one_epoc()
            test_loss, test_acc = self.test((self.test_loader, 'Test'))
            train_loss, train_acc = self.test((self.train_loader, 'train'))
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            print("Valid loss = {} \t Valid accuracy = {}%".format(test_loss, test_acc))

        self.write_test_pred(test_acc)
        confusion_matrix_creator(self.model, self.test_loader)
        print("Train_acc = {}\nTrain_loss = {}\nTest_acc = {}\nTest_loss = {}".format(train_acc_list, train_loss_list, test_acc_list, test_loss_list))

    def write_test_pred(self, accuracy):
        self.model.eval()
        predictions = []
        test = pickle.load(open('test.pickle', 'rb'))
        for data in test:
            image = Variable(data).to(device)
            output = self.model(image)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions.append(str(pred.item()))
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        with open('E:\\DataScienceCourse\\test{}.txt'.format(accuracy), 'w') as f:
            f.write('\n'.join(predictions))


class ResnetTransferTrainer(Trainer):
    def __init__(self, batch_size, num_epocs, critetion, optimizer):
        super(ResnetTransferTrainer, self).__init__()
        self.batch_size = batch_size
        loaders, self.model = tavi_resnet(batch_size)
        self.model = self.model.to(device)
        self.train_loader, self.test_loader = loaders
        self.criterion = critetion
        if optimizer is 'SGD':
            self.optimizer = SGD(params=self.model.fc.parameters(), lr=0.01)
        else:
            self.optimizer = Adam(params=self.model.fc.parameters(), lr=0.002)
        self.epoc_num = num_epocs

    def train_one_epoc(self):
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(device), labels.to(device)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = self.model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = self.criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            self.optimizer.step()

    def test(self, to_test_by):
        to_test_by_loader, to_test_by_name = to_test_by
        self.model.eval()
        test_loss = 0.0
        correct = 0.0
        for data, target in to_test_by_loader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()
        len_data = len(to_test_by_loader) if to_test_by_name is not 'train' else len(
            to_test_by_loader) * self.batch_size
        test_loss /= len_data
        print("Info: correct {} / {}".format(correct, len_data))
        return test_loss, float(100.0 * correct) / len_data

    def train(self, dictionary=None):
        test_loss_list, test_acc_list = [], []
        train_acc_list, train_loss_list = [], []
        for x in range(1, self.epoc_num + 1):
            print("Epoc # {}".format(x))
            self.train_one_epoc()
            test_loss, test_acc = self.test((self.test_loader, 'Test'))
            train_loss, train_acc = self.test((self.train_loader, 'train'))
            print("Valid loss: {} \t Valid accuracy: {}%".format(test_loss, test_acc))
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
        confusion_matrix_creator(self.model, self.test_loader)
        print("Train_acc = {}\nTrain_loss = {}\nTest_acc = {}\nTest_loss = {}".format(train_acc_list, train_loss_list,
                                                                                      test_acc_list, test_loss_list))

    def write_test_pred(self, accuracy):
        self.model.eval()
        predictions = []
        test = pickle.load(open('test.pickle', 'rb'))
        for data in test:
            image = Variable(data)
            image = image.to(device)
            output = self.model(image)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions.append(str(pred.item()))
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        with open('test__{}____{}__.pred'.format(st, accuracy), 'w') as f:
            f.write('\n'.join(predictions))


def single():
    model = ConvulutionalNeuralNetwork(50, 150, 512, 256).to(device)
    batch_size = 200
    LR = 0.3
    num_epocs = 15
    res = {}
    trainer = TrainerModels(model, num_epocs, get_data_loaders_cifar10(batch_size), nn.CrossEntropyLoss(),
                            SGD(params=model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4), 0, batch_size)
    trainer.train(dictionary=res)


def resenet():
    trainer = ResnetTransferTrainer(batch_size=64, num_epocs=5, critetion=nn.CrossEntropyLoss(), optimizer='SGD')
    trainer.train()


def parse_table(df):
    models = []
    model = None
    for index, row in df.iterrows():
        batch_size = int(row[0])
        LR = float(row[1])
        num_epocs = int(row[2])
        channel_in1 = int(row[3])
        channel_out1 = int(row[4])
        hidden_size1 = int(row[5])
        hidden_size2 = int(row[6])
        model = ConvulutionalNeuralNetwork(channel_in1, channel_out1, hidden_size1, hidden_size2).to(device)
        criterion = nn.CrossEntropyLoss()
        if row[7] in ["adam", "Adam", "ADAM"]:
            optimizer = Adam(model.parameters(), lr=LR)
        else:
            optimizer = SGD(model.parameters(), lr=LR)
        models.append(TrainerModels(model, num_epocs, get_data_loaders_cifar10(batch_size), criterion, optimizer, index,
                                    batch_size))
    return models


def multi():
    trainers = parse_table(pd.read_csv('./models.csv'))
    res = {"model_no": [], "epoch_no": [], "valid_acc": [], "valid_loss": []}
    for i, trainer in enumerate(trainers):
        print("Proccessing {} / {}".format(i + 1, len(trainers)))
        trainer.train(dictionary=res)
    df2 = pd.DataFrame(data=res)
    df2.to_csv(path_or_buf="./results.csv")


resenet()
