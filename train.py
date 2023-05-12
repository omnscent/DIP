import torch
from torch import nn

def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        compare = y_hat.type(y.dtype) == y
        return float(compare.type(y.dtype).sum())

def evaluate_accuracy(net, data, device):
        acc_sum = 0
        sam_sum = 0
        if isinstance(net, torch.nn.Module):
            net.eval()
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                acc = accuracy(y_hat, y)
                acc_sum += float(acc)
                sam_sum += y.numel()
        return float(acc_sum / sam_sum)

def train_epoch(net, train_iter, loss, updater, device):
        acc_sum = 0
        loss_sum = 0
        sam_sum = 0
        if isinstance(net, torch.nn.Module):
            net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            acc = accuracy(y_hat, y)
            loss_sum += float(l.sum())
            acc_sum += float(acc)
            sam_sum += y.numel()
        return acc_sum / sam_sum, loss_sum / sam_sum

def train(net, train_iter, test_iter, num_epochs, loss, updater, device):
        train_acc = []
        train_loss = []
        test_acc = []
        for epoch in range(num_epochs):
            train_res = train_epoch(net, train_iter, loss, updater, device)
            test_res = evaluate_accuracy(net, test_iter, device)
            train_acc = train_acc + [train_res[0]]
            train_loss = train_loss + [train_res[1]]
            test_acc = test_acc + [test_res]
            print("epoch = ",
                epoch + 1,
                ", train_acc = ",
                train_acc[-1],
                ", train_loss = ",
                train_loss[-1],
                ", test_acc = ",
                test_acc[-1],
            )
        return train_acc, train_loss, test_acc