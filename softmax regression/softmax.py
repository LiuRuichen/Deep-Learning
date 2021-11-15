import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

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

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.01)

def init_bias(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, val = 0) 
        
num_inputs = 784
num_outputs = 10

model = torch.nn.Sequential(

    nn.Flatten(),
    nn.Linear(num_inputs, num_outputs))

net = model

net.apply(init_weights)
net.apply(init_bias)

# 定义损失函数，包括softmax运算和交叉熵损失计算
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()            
        n += y.shape[0]
    
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad() # 这里我们用到优化器，所以直接对优化器行梯度清零
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # 用到优化器这里

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0] 
                    
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

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
    '''
    首先进行可视化
    '''
    train_iter, test_iter = loaddata(18)
    X, y = next(iter(train_iter))
    visualizing(X.reshape(18, 28, 28), 2, 9, titles = get_fashion_mnist_labels(y));
    
    '''
    规定参数
    '''
    batch_size = 256
    num_epoches = 5
    
    '''
    训练网络
    '''
    train_iter, test_iter = loaddata(batch_size)
    train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, optimizer)
    predict_ch3(net, test_iter)
    
if __name__ == '__main__':
    main()