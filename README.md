# 数字图像处理大作业代码

## 文件结构

`main.py`是主程序，用来执行代码；`Data_Loader.py`用来加载数据集（MNIST，Fashion MNIST和CIFAR-10），`Net.py`为模型设置，`train.py`为训练函数，`Optim.py`只存放了一个手搓的`SGD`函数。

## 使用方法

执行`main`程序就可以。

在`main`当中，可以设置训练的参数，包括小批量数量`batch_size`，轮数`num_epochs`，学习率`lr`，损失函数`loss`，训练用的数据`train_iter, test_iter`。

提供的模型有：感知机（单层和多层的、自己写的和PyTorch提供的都有），LeNet，AlexNet，VGG11，NiN12，GoogLeNet，ResNet18，DenseNet121，HTNet。

**使用要点：**

1. 如果是用自己写的感知机，需要通过`net.train`的方式训练，即

   ```python
   train_acc, train_loss, test_acc = net.train(
       train_iter, test_iter, num_epochs, lr, "default"
   )
   ```

   此时可以使用自己写的`SGD`函数，也可以通过参数`"default"`来使用PyTorch提供的SGD函数。如果用其他的模型，则使用

   ```python
   net.to(device)
   trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
   train_acc, train_loss, test_acc = train.train(
       net, train_iter, test_iter, num_epochs, loss, trainer, device
   )
   ```

   注意，两个部分不能同时出现。可以设置`device`来部署模型，可选项有

   ```python
   device = torch.device("mps")	# Apple Silicon系列SoC
   device = torch.device("cuda")	# 支持CUDA的GPU
   device = torch.device("cpu")	# CPU
   ```

2. 每个模型的训练参数（学习率等）不完全相同，参数见文章。
3. 对于部分模型，需要通过指定数据集读取函数中的`resize`参数来修改大小，参数见文章。
4. 对于CIFAR-10数据集，需要修改部分模型的输入通道数为3。
