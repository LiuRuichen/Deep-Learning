import torch
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader_workers():  
    """使用4个进程来读取数据。"""
    return 4

def loaddata(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root = r'data', 
                                                train = True, 
                                                download = False, 
                                                transform = transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(root = r'data', 
                                               train = False, 
                                               download = False, 
                                               transform = transforms.ToTensor())
    
    train_iter = torch.utils.data.DataLoader(mnist_train, 
                                         batch_size = batch_size, 
                                         shuffle = True, 
                                         num_workers = get_dataloader_workers())

    test_iter = torch.utils.data.DataLoader(mnist_test, 
                                        batch_size = batch_size, 
                                        shuffle = False, 
                                        num_workers = get_dataloader_workers())
    
    
    return train_iter, test_iter


def get_fashion_mnist_labels(labels): 
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        _, predicted = torch.max(y_hat, dim = 1) #y_hat.shape = [256, 10]
        acc_sum += (predicted == y).sum().item()
        n += y.shape[0]
    return acc_sum / n

LeNet = nn.Sequential(nn.Conv2d(in_channels = 1,
                                out_channels = 6,
                                kernel_size = 5,
                                padding = 2,
                                stride = 1),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size = 2,
                                   stride = 2),
                      nn.Conv2d(in_channels = 6,
                                out_channels = 16,
                                kernel_size = 5,
                                padding = 0,
                                stride = 1),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size = 2,
                                   stride = 2),
                      nn.Flatten(),
                      nn.Linear(400, 120),
                      nn.ReLU(),
                      nn.Linear(120, 84),
                      nn.ReLU(),
                      nn.Linear(84, 10))

def visualizing(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.cpu().numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def predict_ch3(net, test_iter, n = 6):  
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        X = X.to(device)
        y = y.to(device)
    
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis = 1))
    #titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    titles = [str(yy.cpu()) + '\n' + true + '\n' + pred for yy, true, pred in zip(y, trues, preds)]
    visualizing(
        X[0:n].reshape((n, 28, 28)), 1, n, titles = titles[0:n])
    

def main():
    net = LeNet
    net.to(device)
    
    batch_size = 256
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    num_epochs = 10
    learning_rate = 0.2
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate) #梯度下降法求损失函数最小值
    
    train_iter, test_iter = loaddata(batch_size)
    
    tr_accu = np.zeros(num_epochs)
    ts_accu = np.zeros(num_epochs)
    loss_arr = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        sum_loss, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter: # X.shape = torch.Size([256, 1, 28, 28]), y.shape = torch.Size([256])
            X, y = X.to(device), y.to(device)
            y_hat = net(X) # y_hat.shape = torch.Size([256, 10])
            
            l = loss(y_hat, y).sum()           
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()
            
            sum_loss += l.item()
            _, predicted = torch.max(y_hat, dim = 1) #y_hat.shape = [256, 10]
            train_acc_sum += (predicted == y).sum().item()
            n += y.shape[0]
        avg_acc = train_acc_sum / n
        tr_accu[epoch] = avg_acc
        avg_loss = sum_loss / n
        loss_arr[epoch] = avg_loss
        test_acc = evaluate_accuracy(test_iter, net)
        ts_accu[epoch] = test_acc
        
        print('epoch ' + str(epoch + 1) + 
              ', average loss: ' + str(avg_loss) + 
              ', training accuracy: ' + str(avg_acc) + 
              ', test accuracy: ' + str(test_acc))
    
    epochs = np.arange(1, num_epochs + 1)
    fig = plt.figure()
    plt.plot(epochs, tr_accu, linestyle = '--', color = 'cyan', marker = 'o', label = 'training accuracy')
    plt.plot(epochs, ts_accu, linestyle = '-', color = 'red', linewidth = 3, marker = 'o', label = 'test accuracy') 
    plt.legend(loc = 4)
    
    fig = plt.figure()
    plt.plot(epochs, loss_arr, linestyle = '-', color = 'black', linewidth = 3, marker = 'o', label = 'loss function')
    plt.legend(loc = 1)
        
    predict_ch3(net, test_iter, n = 6)
    
    
if __name__ == '__main__':
    main()