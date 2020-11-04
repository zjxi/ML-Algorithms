"""
    Implementation of the logistic regression and maximum entropy model,
    written by zjxi @ 2020/11/3
"""

import numpy as np


def load_data(filename):
    """
    导入训练集或者测试集数据
    :param filename:
    :return:
    """
    feat, label = [], []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split(',')
            # 添加标签向量
            label.append(int(line[-1]))
            # 添加特征向量
            feat.append(int(e) for e in line[:-1])

    return feat, label


def logistic_regression(x):
    x = np.mat(x)
    w = np.zeros((1, len(x)))
    b = 0
    P_y_x1 = np.exp(w * x + b) / (1 + np.exp(w * x + b))
    P_y_x0 = 1 / (1 + np.exp(w * x + b))
    if P_y_x1 > P_y_x0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")
    # 二项逻辑回归
    logistic_regression(train_data, train_lbl)