import torch
import Net
import Optim
import train
from torch import nn
from Data_Loader import *

"""
预设定参数
"""

batch_size = 256
num_epochs = 50
lr = 0.1
# train_iter, test_iter = load_mnist_data(batch_size, resize=224)
train_iter, test_iter = load_Fashion_mnist_data(batch_size, resize=32)
# train_iter, test_iter = load_CIFAR_data(batch_size, resize=224)
loss = nn.CrossEntropyLoss(reduction="none")
device = torch.device("mps")
# device = torch.device("cuda")


"""
模型设置
"""

# net = Net.Linear_Model(784, 10)
# net = Net.Multi_Linear_Model(1, 784, [256], 10)
# net = Net.Multilayer_Perceptron(3, [784, 512, 256, 64, 10])
# net = Net.LeNet()
# net = Net.AlexNet()
# net = Net.VGG11()
# net = Net.NiN12()
# net = Net.GoogLeNet()
# net = Net.ResNet18(1, 10)
# net = Net.DenseNet121()
net = Net.HTNet(1, 10)


"""
训练设置
"""

# train_acc, train_loss, test_acc = net.train(
#     train_iter, test_iter, num_epochs, lr, "default"
# )

net.to(device)
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
train_acc, train_loss, test_acc = train.train(
    net, train_iter, test_iter, num_epochs, loss, trainer, device
)

print(train_acc)
print(train_loss)
print(test_acc)
