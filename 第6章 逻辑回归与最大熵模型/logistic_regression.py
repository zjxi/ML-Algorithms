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
            label.append(line[-1])
            # 添加特征向量
            feat.append(line[:-1])

    return feat, label
