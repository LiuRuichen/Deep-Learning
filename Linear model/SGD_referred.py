import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch import nn
import torch

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


def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

def main (): 
    font1 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 20)
    font2 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 10)
    
    w = torch.tensor([[2, -3.4]]).double()
    w = w.T
    b = 4.2
    features, labels = synthetic_data (w, b, 1000)         
    batch_size = 8
    data_iter = load_array((features, labels), batch_size)
    
    net = nn.Sequential(nn.Linear(2, 1))
    net = net.double()
    net[0].weight.data = torch.full([1, 2], 0).double()
    print(net[0].weight.data)
    net[0].bias.data.fill_(0)
    
    loss = nn.MSELoss()
    
    trainer = torch.optim.SGD(net.parameters(), lr = 0.005)
    
    plt.scatter(features[:,1], labels[:,0], marker = 'o', color = 'slategrey', s = 1, label = '数据集可视化')
    plt.title('数据集可视化', fontproperties = font1)
    xx = np.linspace(-3, 3, 1000)
    A = np.full((1000, 2), 1.)
    A[:,1] = xx
    A = torch.tensor(A)
    
    num_epochs = 3
    for epoch in range(num_epochs):            
        for X, y in data_iter:
            print(X.shape)
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        
        A1 = A
        with torch.no_grad():  
            if epoch == 0:
                plt.plot(xx, net(A1).numpy()[:,0], color = 'red', label = '第' + str(epoch + 1) + '次扫描结果')
            elif epoch == 1:
                plt.plot(xx, net(A1).numpy()[:,0], color = 'black', label = '第' + str(epoch + 1) + '次扫描结果')
            else:
                plt.plot(xx, net(A1).numpy()[:,0], color = 'cyan', label = '第' + str(epoch + 1) + '次扫描结果')
            
        print(net[0].weight.data)
        l = loss(net(features), labels)
        print('epoch ' + str(epoch + 1) + ', loss: ' + str(float(l.mean())))
    
    plt.legend(loc = 3, prop = font2)

if __name__ == '__main__':
    main()