# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:14:26 2021
"""
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch import nn
import torch

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)

def init_bias(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, val = 0)

def regul(X):
    X1 = torch.cat([torch.full([X.shape[0], 1], 1.), X], dim = 1)
    X2 = torch.cat([X1, X ** 2], dim = 1)
    X3 = torch.cat([X2, X ** 3], dim = 1)
    return X3
    
def loss_on_test(data_iter, net, loss):
    test_l_sum, n = 0, 0
    for X, y in data_iter:
        X = regul(X)
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        test_l_sum += l.item()           
        n += y.shape[0] 
    return test_l_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    print('------------------This is cubic fitting--------------------')
    for epoch in range(num_epochs):
        train_l_sum, n = 0.0, 0
        for X, y in train_iter:  #X的大小就是batch_size，循环次数就是sample_num/batch_size
            X = regul(X)
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
            n += y.shape[0] 
                    
        loss_ = train_l_sum / n
        loss_2 = loss_on_test(test_iter, net, loss)
        print('epoch %d, loss on training is %.4f while loss on testing is %.4f' % (epoch + 1, loss_, loss_2))
      
def train (train_features, train_labels, test_features, test_labels, batch_size):
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size)
    net = nn.Sequential(
        nn.Linear(4, 1))
    net = net.double()
    net.apply(init_weights)
    net.apply(init_bias)
    
    loss = nn.MSELoss()   
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    
    num_epochs = 10
    
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
    
    font1 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 13)
    font2 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 20)
    with torch.no_grad():  
        plt.figure()
        plt.scatter(train_features[:,0], train_labels, s = 3, c = 'slategrey', label = '样本点')
        '''
        train_features并不是排好序的，直接绘图会错乱
        '''
        start = torch.min(train_features)
        end = torch.max(train_features)
        xx = torch.linspace(start, end, 1000).double()  #必须做类型转换
        xx_= xx.reshape(-1, 1)
        plt.plot(xx, net(regul(xx_)), color = 'red', linewidth = 1, label = '三次拟合曲线（最佳拟合）')
        plt.legend(loc = 4, prop = font1)
        plt.title('最佳拟合图示', fontproperties = font2)