# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:55:27 2021
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import random
import math


max_degree = 20  # 多项式的最大阶数
n_train, n_test = 5, 5  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间,(20,)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size = (n_train + n_test, 1)) #平均值0, 标准差, 200*1
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) #(200, 20)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n) = (n-1)!
    
labels = np.dot(poly_features, true_w) #(200,)
labels += np.random.normal(scale = 0.1, size = labels.shape) #噪声项

def sort_arr (X):
    return np.sort(X)

X = features[:,0]
Y = labels

def seq_list(X, Y): #将X排序，且Y随X的索引变化
    Y1 = np.zeros(len(Y),)
    seq = np.argsort(X)
    for i in range(0, len(Y)):
        Y1[i] = Y[seq[i]]
    X1 = np.sort(X)
    return X1, Y1
    
idx = np.arange(0, len(Y))
random.shuffle(idx)

X_train = X[idx[0:n_train]]
Y_train = Y[idx[0:n_train]]

X_test = X[idx[n_train+1:]]
Y_test = Y[idx[n_train+1:]]

font1 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 13)
font2 = font_manager.FontProperties(fname = r'C:\Users\未央\AppData\Local\Microsoft\Windows\Fonts\方正楷体_GBK.TTF', size = 20)
    
plt.scatter(X_train, Y_train, s = 5, c = 'slategrey', label = '训练样本点')
plt.scatter(X_test, Y_test, s = 5, c = 'black', label = '测试样本点')

X_train, Y_train = seq_list(X_train, Y_train)
X_test, Y_test = seq_list(X_test, Y_test)

x_train = X_train.reshape(-1, 1)
y_train = Y_train.reshape(-1, 1)
x_test = X_test.reshape(-1, 1)
y_test = Y_test.reshape(-1, 1)

print('------------------This is High order fitting--------------------')
ploy_reg = PolynomialFeatures(degree = 9)
X_ploy = ploy_reg.fit_transform(x_train)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_ploy,y_train)
params = lin_reg.coef_

loss = lin_reg.predict(ploy_reg.fit_transform(x_train)) - y_train
loss_ = lin_reg.predict(ploy_reg.fit_transform(x_test)) - y_test

print('loss on training is %.4f while loss on testing is %.4f' % (loss.mean(axis = 0), abs(loss_.mean(axis = 0))))

X = np.sort(X)
X = X.reshape(-1, 1)
plt.plot(X, lin_reg.predict(ploy_reg.fit_transform(X)), color = 'red', linewidth = 1, label = '高阶拟合曲线（过拟合）')

plt.legend(loc = 4, prop = font1)
plt.title('过拟合图示', fontproperties = font2)