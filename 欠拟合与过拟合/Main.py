import math
import numpy as np
import torch
from fittings.LineFitting import train as train_linear
from fittings.CubicFitting import train as train_cubic

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间,(20,)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size = (n_train + n_test, 1)) #平均值0, 标准差, 200*1
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) #(200, 20)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
    
labels = np.dot(poly_features, true_w) #(200,)
labels += np.random.normal(scale = 0.1, size = labels.shape) #噪声项

true_w = torch.tensor(true_w).double()
features = torch.tensor(features).double()
poly_features = torch.tensor(poly_features).double()
labels = torch.tensor(labels).double()

batch_size = 8

train_linear (features[0:n_train, :], labels[0:n_train], features[n_train + 1:, :], labels[n_train + 1:], batch_size)
train_cubic (features[0:n_train, :], labels[0:n_train], features[n_train + 1:, :], labels[n_train + 1:], batch_size)