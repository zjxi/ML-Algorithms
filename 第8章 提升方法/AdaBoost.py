"""
    Implementation of the AdaBoost,
    written by zjxi @ 2020/11/9
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
            label.append(int(line[-1]))
            # 添加特征向量
            feat.append([float(e) for e in line[:-1]])

    return feat, label


class AdaBoost:
    def __init__(self):
        pass
