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
    def __init__(self, trainData, trainLbl, testData, testLbl, lr, epoch):
        """
        初始化逻辑回归的各类参数
        :param trainData: 训练特征向量集
        :param trainLbl: 训练标签向量集
        :param testData: 测试特征向量集
        :param testLbl: 测试标签向量集
        :param lr: 学习率
        :param epoch: 训练迭代轮数
        """
        self.train_data = trainData
        self.train_label = trainLbl
        self.test_data = testData
        self.test_label = testLbl
        self.epoch = epoch
        # 初始化权重，学习率
        self.w = np.zeros((1, np.shape(self.train_data)[1]))
        self.lr = lr
        # 分别将特征向量、标签向量转化为矩阵并进行相应的转置变化
        self.train_data = np.mat(self.train_data)
        self.train_label = np.mat(self.train_label).T
        self.test_data = np.mat(self.test_data)
        self.test_label = np.mat(self.test_label).T

    def training(self):
        pass

    def test(self):
        pass
    

if __name__ == '__main__':
    pass
