import torch
from torch import nn
from torch.nn import functional as F

"""
单层神经网络
"""


class Linear_Model:
    def __init__(self, num_of_inputs, num_of_outputs):
        self.W = nn.Parameter(
            torch.randn(num_of_inputs, num_of_outputs, requires_grad=True) * 0.01
        )
        self.b = nn.Parameter(torch.zeros(num_of_outputs, requires_grad=True))
        self.params = [self.W, self.b]

    def softmax(self, X):
        Exp = torch.exp(X)
        Sum = Exp.sum(1, keepdim=True)
        return Exp / Sum

    def cross_entropy_loss(self, y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    def net(self, X):
        return self.softmax(
            torch.matmul(X.reshape(-1, self.W.shape[0]), self.W) + self.b
        )

    def accuracy(self, y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        compare = y_hat.type(y.dtype) == y
        return float(compare.type(y.dtype).sum())

    def evaluate_accuracy(self, data):
        accu_sum = 0
        sam_sum = 0
        with torch.no_grad():
            for X, y in data:
                y_hat = self.net(X)
                acu = self.accuracy(y_hat, y)
                accu_sum += float(acu)
                sam_sum += y.numel()
        return float(accu_sum / sam_sum)

    def train_epoch(self, train_iter, lr, updater):
        accu_sum = 0
        loss_sum = 0
        sam_sum = 0
        for X, y in train_iter:
            y_hat = self.net(X)
            l = self.cross_entropy_loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater([self.W, self.b], lr, X.shape[0])
            acu = self.accuracy(y_hat, y)
            loss_sum += float(l.sum())
            accu_sum += float(acu)
            sam_sum += y.numel()
        return accu_sum / sam_sum, loss_sum / sam_sum

    def train(self, train_iter, test_iter, num_epochs, lr, updater):
        train_acu = []
        train_loss = []
        test_acu = []
        if updater == "default":
            updater = torch.optim.SGD(self.params, lr)
        for epoch in range(num_epochs):
            train_res = self.train_epoch(train_iter, lr, updater)
            test_res = self.evaluate_accuracy(test_iter)
            train_acu = train_acu + [train_res[0]]
            train_loss = train_loss + [train_res[1]]
            test_acu = test_acu + [test_res]
            print(
                "epoch = ",
                epoch + 1,
                ", train_acc = ",
                train_acu[-1],
                ", train_loss = ",
                train_loss[-1],
                ", test_acc = ",
                test_acu[-1],
            )
        return train_acu, train_loss, test_acu


"""
双层神经网络
"""


class Multi_Linear_Model:
    def __init__(
        self, num_of_hidden_layers, num_of_inputs, hidden_layers, num_of_outputs
    ):
        self.num_of_hidden_layers = num_of_hidden_layers
        self.W = []
        self.b = []
        self.params = []
        self.W += [
            nn.Parameter(
                torch.randn(num_of_inputs, hidden_layers[0], requires_grad=True)
            )
        ]
        self.params += [self.W[-1]]
        self.b += [nn.Parameter(torch.zeros(hidden_layers[0], requires_grad=True))]
        self.params += [self.b[-1]]
        for layers in range(num_of_hidden_layers - 1):
            self.W += [
                nn.Parameter(
                    torch.randn(
                        hidden_layers[layers],
                        hidden_layers[layers + 1],
                        requires_grad=True,
                    )
                )
            ]
            self.params += [self.W[-1]]
            self.b += [
                nn.Parameter(torch.zeros(hidden_layers[layers + 1], requires_grad=True))
            ]
            self.params += [self.b[-1]]
        self.W += [
            nn.Parameter(
                torch.randn(hidden_layers[-1], num_of_outputs, requires_grad=True)
            )
        ]
        self.params += [self.W[-1]]
        self.b += [nn.Parameter(torch.zeros(num_of_outputs, requires_grad=True))]
        self.params += [self.b[-1]]

    def softmax(self, X):
        Exp = torch.exp(X)
        Sum = Exp.sum(1, keepdim=True)
        return Exp / Sum

    def ReLU(self, X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    def net(self, X):
        temp = X.reshape(-1, self.W[0].shape[0])
        for layers in range(self.num_of_hidden_layers):
            temp = self.ReLU(torch.matmul(temp, self.W[layers]) + self.b[layers])
        temp = torch.matmul(temp, self.W[-1]) + self.b[-1]
        return temp

    def accuracy(self, y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        compare = y_hat.type(y.dtype) == y
        return float(compare.type(y.dtype).sum())

    def evaluate_accuracy(self, data):
        accu_sum = 0
        sam_sum = 0
        with torch.no_grad():
            for X, y in data:
                y_hat = self.net(X)
                acu = self.accuracy(y_hat, y)
                accu_sum += float(acu)
                sam_sum += y.numel()
        return float(accu_sum / sam_sum)

    def train_epoch(self, train_iter, lr, updater):
        accu_sum = 0
        loss_sum = 0
        sam_sum = 0
        for X, y in train_iter:
            y_hat = self.net(X)
            loss = nn.CrossEntropyLoss(reduction="none")
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater([self.W, self.b], lr, X.shape[0])
            acu = self.accuracy(y_hat, y)
            loss_sum += float(l.sum())
            accu_sum += float(acu)
            sam_sum += y.numel()
        return accu_sum / sam_sum, loss_sum / sam_sum

    def train(self, train_iter, test_iter, num_epochs, lr, updater):
        train_acu = []
        train_loss = []
        test_acu = []
        if updater == "default":
            updater = torch.optim.SGD(self.params, lr)
        for epoch in range(num_epochs):
            train_res = self.train_epoch(train_iter, lr, updater)
            test_res = self.evaluate_accuracy(test_iter)
            train_acu = train_acu + [train_res[0]]
            train_loss = train_loss + [train_res[1]]
            test_acu = test_acu + [test_res]
            print(
                "epoch = ",
                epoch + 1,
                ", train_acc = ",
                train_acu[-1],
                ", train_loss = ",
                train_loss[-1],
                ", test_acc = ",
                test_acu[-1],
            )
        return train_acu, train_loss, test_acu


"""
多层感知机
"""


def Multilayer_Perceptron(hidden_num, inputs):
    layers = []
    layers.append(nn.Flatten())
    for i in range(hidden_num):
        layers.append(nn.Linear(inputs[i], inputs[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(inputs[-2], inputs[-1]))
    net = nn.Sequential(*layers)
    for w in net:
        if type(w) == nn.Linear:
            nn.init.normal_(w.weight, std=0.1)
    return net


"""
LeNet
"""


def LeNet():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 6 * 6, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )
    for w in net:
        if type(w) == nn.Linear or type(w) == nn.Conv2d:
            nn.init.xavier_uniform_(w.weight)
    return net


"""
AlexNet
"""


def AlexNet(input_chann_num, output_chann_num):
    net = nn.Sequential(
        nn.Conv2d(input_chann_num, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, output_chann_num),
    )
    for w in net:
        if type(w) == nn.Linear or type(w) == nn.Conv2d:
            nn.init.xavier_uniform_(w.weight)
    return net


"""
VGG
"""


def VGG_Block(conv_num, input_chann_num, output_chann_num):
    layers = []
    for i in range(conv_num):
        layers.append(
            nn.Conv2d(input_chann_num, output_chann_num, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        input_chann_num = output_chann_num
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


def VGG(arg_num):
    layers = []
    input_chann_num = 1
    for [conv_num, output_chann_num] in arg_num:
        layers = layers + VGG_Block(conv_num, input_chann_num, output_chann_num)
        input_chann_num = output_chann_num
    layers = layers + [
        nn.Flatten(),
        nn.Linear(output_chann_num * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10),
    ]
    net = nn.Sequential(*layers)
    for w in net:
        if type(w) == nn.Linear or type(w) == nn.Conv2d:
            nn.init.xavier_uniform_(w.weight)
    return net


def VGG11():
    arg_num = [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]]
    return VGG(arg_num)


"""
NiN
"""


def NiN_Block(input_chann_num, output_chann_num, kernel_size, strides, padding):
    conv1 = nn.Conv2d(input_chann_num, output_chann_num, kernel_size, strides, padding)
    conv2 = nn.Conv2d(output_chann_num, output_chann_num, kernel_size=1)
    conv3 = nn.Conv2d(output_chann_num, output_chann_num, kernel_size=1)
    block = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU())
    return block


def NiN12():
    net = nn.Sequential(
        NiN_Block(1, 96, 11, 4, 0),
        nn.MaxPool2d(3, stride=2),
        NiN_Block(96, 256, 5, 1, 2),
        nn.MaxPool2d(3, stride=2),
        NiN_Block(256, 384, 3, 1, 1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        NiN_Block(384, 10, 3, 1, 1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return net


"""
GoogLeNet
"""


class Inception_Block(nn.Module):
    def __init__(self, input_chann_num, o1, o2, o3, o4):
        super().__init__()
        self.L1 = nn.Conv2d(input_chann_num, o1, kernel_size=1)

        self.L21 = nn.Conv2d(input_chann_num, o2[0], kernel_size=1)
        self.L22 = nn.Conv2d(o2[0], o2[1], kernel_size=3, padding=1)

        self.L31 = nn.Conv2d(input_chann_num, o3[0], kernel_size=1)
        self.L32 = nn.Conv2d(o3[0], o3[1], kernel_size=5, padding=2)

        self.L41 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.L42 = nn.Conv2d(input_chann_num, o4, kernel_size=1)

    def forward(self, X):
        L1 = F.relu(self.L1(X))
        L2 = F.relu(self.L22(F.relu(self.L21(X))))
        L3 = F.relu(self.L32(F.relu(self.L31(X))))
        L4 = F.relu(self.L42(self.L41(X)))
        return torch.cat((L1, L2, L3, L4), dim=1)


def GoogLeNet():
    p1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    p2 = nn.Sequential(
        Inception_Block(192, 64, (96, 128), (16, 32), 32),
        Inception_Block(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    p3 = nn.Sequential(
        Inception_Block(480, 192, (96, 208), (16, 48), 64),
        Inception_Block(512, 160, (112, 224), (24, 64), 64),
        Inception_Block(512, 128, (128, 256), (24, 64), 64),
        Inception_Block(512, 112, (144, 288), (32, 64), 64),
        Inception_Block(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    p4 = nn.Sequential(
        Inception_Block(832, 256, (160, 320), (32, 128), 128),
        Inception_Block(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1024, 10),
    )
    net = nn.Sequential(p1, p2, p3, p4)
    return net


"""
ResNet
"""


class Residual(nn.Module):
    def __init__(self, input_chann_num, output_chann_num, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_chann_num, output_chann_num, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(
            output_chann_num, output_chann_num, kernel_size=3, padding=1
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_chann_num, output_chann_num, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_chann_num)
        self.bn2 = nn.BatchNorm2d(output_chann_num)

    def forward(self, X):
        Y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_chann_num, output_chann_num, residual_num, first_block=False):
    layers = []
    for i in range(residual_num):
        if i == 0 and not first_block:
            layers.append(
                Residual(input_chann_num, output_chann_num, use_1x1conv=True, strides=2)
            )
        else:
            layers.append(Residual(output_chann_num, output_chann_num))
    return layers


def ResNet18(input_chann_num, output_chann_num):
    p1 = nn.Sequential(
        nn.Conv2d(input_chann_num, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    p2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    p3 = nn.Sequential(*resnet_block(64, 128, 2))
    p4 = nn.Sequential(*resnet_block(128, 256, 2))
    p5 = nn.Sequential(*resnet_block(256, 512, 2))
    p6 = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, output_chann_num)
    )
    net = nn.Sequential(p1, p2, p3, p4, p5, p6)
    return net


"""
DenseNet
"""


class DenseBlock(nn.Module):
    def __init__(self, conv_num, input_chann_num, output_chann_num):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(conv_num):
            layer.append(
                nn.BatchNorm2d(output_chann_num * i + input_chann_num),
                nn.ReLU(),
                nn.Conv2d(
                    output_chann_num * i + input_chann_num,
                    output_chann_num,
                    kernel_size=3,
                    padding=1,
                ),
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for w in self.net:
            Y = w(X)
            X = torch.cat((X, Y), dim=1)
        return X


def TransitionBlock(input_chann_num, output_chann_num):
    return nn.Sequential(
        nn.BatchNorm2d(input_chann_num),
        nn.ReLU(),
        nn.Conv2d(input_chann_num, output_chann_num, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


def DenseNet121():
    p1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    p2 = []
    chann_num = 64
    grow_rate = 32
    arg_num = [4, 4, 4, 4]
    for i, conv_num in enumerate(arg_num):
        p2.append(DenseBlock(conv_num, chann_num, grow_rate))
        chann_num += conv_num * grow_rate
        if i != len(arg_num) - 1:
            p2.append(TransitionBlock(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net = nn.Sequential(
        p1,
        *p2,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    )
    return net


"""
HTNet
"""


class HTBlock(nn.Module):
    def __init__(self, input_chann_num, o1, o2, strides=1):
        super().__init__()
        self.conv11 = nn.Conv2d(
            input_chann_num, o1, kernel_size=3, padding=1, stride=strides
        )
        self.conv12 = nn.Conv2d(o1, o1, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(
            input_chann_num, o2, kernel_size=5, padding=2, stride=strides
        )
        self.conv22 = nn.Conv2d(o2, o2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(input_chann_num, o1 + o2, kernel_size=1, stride=strides)

    def forward(self, X):
        Y1 = self.conv12(F.relu(self.conv11(X)))
        Y2 = self.conv22(F.relu(self.conv21(X)))
        Y = torch.cat((Y1, Y2), dim=1)
        X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)


def HTNet(input_chann_num, output_chann_num):
    net = nn.Sequential(
        nn.Conv2d(input_chann_num, 64, kernel_size=7, padding=3, stride=2),
        HTBlock(64, 32, 32, 1),
        HTBlock(64, 64, 64, 1),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        HTBlock(128, 64, 64, 1),
        nn.Conv2d(128, output_chann_num, kernel_size=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return net
