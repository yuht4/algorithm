import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Perceptron import Perceptron

from matplotlib.colors import ListedColormap


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None);

y = df.iloc[0:100, 4].values;  #### 读取Dataframe 第四列 的所有value，形成一个ndarray array([a,b,b,c])

y = np.where(y == "Iris-setosa", -1, 1);  #### 当value 符合 替换成1 否则 -1 array([1,1,1,1,-1,1,1])

X = df.iloc[0:100, [0, 2]].values;  ### 截取两个features array([ [1,2],[3,4],[12,33] ] )返回一个多维数组 ndarray

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa');  ### ndarray 访问 row 和 col

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label = 'versiocolor');

#### 这里 的前五十 和 后五十 本来就是文件里面规定好的
#### Iris-setosa ==> -1
#### Iris-versiocolor ==> 1

plt.xlabel('petal length');
plt.ylabel('sepal length');

plt.legend(loc = 'upper left');

plt.show();


#### vector y 表示的真实的 output
#### Features Matrix X 两个指标，形成[a,b]的 二维array

ppn = Perceptron(0.1, 10);
ppn.fit(X, y);

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_ , marker = 'o');

plt.xlabel('Epochs');
plt.ylabel('Number of miscalssifications');

plt.show();


#### 画出 每一次 epoce 得到的 mismatch个数，注意把所有的sample 都过一边 就是一次epoch

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    markers = ('s', 'x', 'o', '^', 'v');
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan');

    cmap = ListedColormap(colors[:len(np.unique(y))]);
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1;
    
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1;

    #### 生成二维网格矩阵
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T);

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
