import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
import random

def synthetic_data (w, b, num_samples):
    '''
    生成 y = Xw + b + epsilon格式的线性数据集，epsilon是噪声
    '''
    X = torch.from_numpy(np.zeros((num_samples, 2)))   
    X[:,0] = torch.from_numpy(np.ones((num_samples,)))
    X[:,1] = torch.normal(0, 1, (num_samples,)) # 0-1正态分布的随机函数
    b = torch.from_numpy(np.full((num_samples, 1), b)).double() 
    y = torch.matmul(X, w) + b # num_samples * 1 的矩阵
    y += torch.normal(0, 2, y.shape)  #添加噪声项, num_samples * 1 的矩阵
    return X, y


def data_iter (batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)]) #得到一个批量
        yield features[batch_indices], labels[batch_indices]


def squared_loss (y_hat, y):
    return (y_hat - y) ** 2 / 2 #均方误差向量


def sgd (params, learning_rate, batch_size):
    '''
    小批量随机梯度下降
    '''
    with torch.no_grad():
        for params in params:
            params -= learning_rate * params.grad / batch_size
            params.grad.zero_()

    
def linreg (X, w, b):
    X = X.double()
    w = w.double()
    result = torch.matmul(X, w) + b
    return result
    
      
def main ():         
    font = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF')
    w = torch.tensor([[2, -3.4]]).double()
    w = w.T
    w_true = w
    b = 4.2
    b_true = b
    features, labels = synthetic_data (w, b, 1000)
    plt.figure()
    
    plt.subplot(131)
    plt.scatter(features[:,1], labels[:,0], marker = '.', color = 'blue', s = 10)
    
    '''
    初始化模型参数  
    '''
    w = torch.zeros([2, 1], requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    batch_size = 8
    learning_rate = 0.005
    num_epoches = 8
    net = linreg
    loss = squared_loss
    
    xx = np.linspace(-3, 3, 1000)

    A = np.full((1000, 2), 1.)
    A[:,1] = xx
    
    for epoch in range(num_epoches):
        if epoch == 0:
            plt.subplot(263)
        elif epoch == 1:
            plt.subplot(264)
        elif epoch == 2:
            plt.subplot(265)
        elif epoch == 3:
            plt.subplot(266)
        elif epoch == 4:
            plt.subplot(269)
        elif epoch == 5:
            plt.subplot(2,6,10)
        elif epoch == 6:
            plt.subplot(2,6,11)
        elif epoch == 7:
            plt.subplot(2,6,12)
        plt.scatter(features[:,1], labels[:,0], marker = '.', color = 'blue', s = 5)
        
        w_true1 = w_true.numpy()
        plt.plot(xx, np.dot(A, w_true1) + b_true, linewidth = 2, color = 'red')
        
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()        
            sgd([w, b], learning_rate, batch_size)
            w_tmp = w
            b_tmp = b
            with torch.no_grad():  
                '''
                直接w.numpy()会破坏计算图，我们用一个变量来代替它
                或者我们可以用detach方法
                w_tmp = w_tmp.detach().numpy() 
                b_tmp = b_tmp.detach().numpy()
                '''
                w_tmp = w_tmp.numpy()
                b_tmp = b_tmp.numpy()

                plt.plot(xx, np.dot(A, w_tmp) + b_tmp, linewidth = 0.5)
                
            plt.title("第" + str(epoch + 1) + '次扫描', fontproperties = font)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print('epoch ' + str(epoch + 1) + ', loss: ' + str(float(train_l.mean())))
        
        plt.tight_layout()
    
    '''
    plt.figure()
    x = np.linspace(-3, 3, 1000)
    y = -3.4 * x + 4.2
    plt.plot(x, y, 'r')
    plt.scatter(features[:,1], labels[:,0], c = 'b')
    '''
        
  
if __name__ == '__main__':
    main()