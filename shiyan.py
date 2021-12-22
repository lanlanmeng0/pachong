import numpy as np
import pylab as pl  # 绘图功能
from sklearn import svm

# 创建 200 个点
np.random.seed(0) # 让每次运行程序生成的随机样本点不变
X = np.r_[np.random.randn(100, 2) - [2, 2], np.random.randn(100, 2) + [2, 2]]
# 两个类别 每类有 100个点，Y 为 200 行 1 列的列向量
Y = [0] * 100 + [1] * 100

# 建立 svm 模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 点斜式：y = -(w0/w1)x - (w2/w1)
w = clf.coef_[0]  
a = -w[0] / w[1]  # 斜率
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程

# 画出和划分超平面平行且经过支持向量的两条线（斜率相同，截距不同）
b = clf.support_vectors_[0] # 取出第一个支持向量点
yy_down = a * xx + (b[1] - a * b[0]) 
b = clf.support_vectors_[-1] # 取出最后一个支持向量点
yy_up = a * xx + (b[1] - a * b[0])

# 查看相关的参数值
print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)
print("clf.coef_: ", clf.coef_)


# 绘制划分超平面，边际平面和样本点
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
# 圈出支持向量
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()