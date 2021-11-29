import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

def get_dataloader_workers():  
    """使用4个进程来读取数据。"""
    return 4

def loaddata(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root = r'.\data', 
                                                train = True, 
                                                download = False, 
                                                transform = transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(root = r'.\data', 
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

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)

def init_bias(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, val = 0)
        
def visualizing(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()            
        n += y.shape[0]
        
    return acc_sum / n
        
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    tr_accu = []
    ts_accu = []
    loss_arr = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        c = 0
        for X, y in train_iter:
            #print(X.shape)
            #print(y.shape)
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad() # 这里我们用到优化器，所以直接对优化器行梯度清零
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            
            optimizer.step()  # 用到优化器这里

            train_l_sum += l.item()
            #print('y_hat:', y_hat.shape)
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0] 
            c += 1
        
        print(c)
                    
        test_acc = evaluate_accuracy(test_iter, net)
        
        train_acc = train_acc_sum / n
        loss_ = train_l_sum / n
        tr_accu.append(train_acc)
        ts_accu.append(test_acc)
        loss_arr.append(loss_)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, loss_, train_acc, test_acc))
    
    return np.array(tr_accu), np.array(ts_accu), np.array(loss_arr)
        
        
def predict_ch3(net, test_iter, n = 6):  
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    visualizing(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def main():
    model = nn.Sequential(
    nn.Flatten(),  # 展平层，pytorch不会隐式地展平
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10))

    net = model

    net.apply(init_weights)
    net.apply(init_bias)
    
    batch_size = 256
    num_epoches = 5
    learning_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    
    '''
    训练网络
    '''
    train_iter, test_iter = loaddata(batch_size)
    loss = nn.CrossEntropyLoss()
    tr_accu, ts_accu, loss_arr = train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, optimizer)
    xr = np.arange(1, num_epoches + 1)
    
    font = font_manager.FontProperties(fname = r'C:\Windows\Fonts\STKAITI.TTF', size = 10)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(xr, tr_accu, linestyle = '--', color = 'cyan', marker = 'o', label = '模型基于训练集的准确率')
    ax1.plot(xr, ts_accu, linestyle = '-', color = 'red', linewidth = 3, marker = 'o', label = '模型基于测试集的准确率') 
    ax1.legend(loc = 4, prop = font)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(xr, loss_arr, linestyle = '-', color = 'black', linewidth = 3, marker = 'o', label = '损失函数曲线')
    ax2.legend(loc = 1, prop = font)
     
    predict_ch3(net, test_iter)
    
if __name__ == '__main__':
    main()