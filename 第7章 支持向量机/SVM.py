"""
    Implementation of the Support Vector Machine (SVM),
    written by zjxi @ 2020/11/8
"""

import numpy as np


def load_data(filename):
    """
    导入训练集或者测试集数据
    :param filename: 数据文件路径
    :return: 特征向量，标签向量
    """
    feat, label = [], []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split(',')
            # 添加标签向量
            label.append(1 if int(line[-1]) == 1 else -1)
            # 添加特征向量
            feat.append([float(e) for e in line[:-1]])

    return feat, label


class LinearSVM:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, c=1, lr=0.01, epoch=10000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        for _ in range(epoch):
            self._w *= 1 - lr
            err = 1 - y * self.predict(x, True)
            idx = np.argmax(err)
            # 注意即使所有 x, y 都满足 w·x + b >= 1
            # 由于损失里面有一个 w 的模长平方
            # 所以仍然不能终止训练，只能截断当前的梯度下降
            if err[idx] <= 0:
                continue
            delta = lr * c * y[idx]
            self._w += delta * x[idx]
            self._b += delta

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self._w) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)


if __name__ == '__main__':
