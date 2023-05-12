import torchvision
from torch.utils import data
from torchvision import transforms


def get_dataloader_workers():
    return 0


def load_mnist_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )


def load_Fashion_mnist_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )


def load_CIFAR_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    CIFAR_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=trans, download=True
    )
    CIFAR_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            CIFAR_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            CIFAR_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )
