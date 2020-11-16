"""
    Implementation of the Expectation Maximum algorithm (EM),
    written by zjxi @ 2020/11/14
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


class EM:
    def __init__(self, trainData, trainLbl, testData, testLbl, epoch):
        """
        初始化逻辑回归的各类参数
        :param trainData: 训练特征向量集
        :param trainLbl: 训练标签向量集
        :param testData: 测试特征向量集
        :param testLbl: 测试标签向量集
        :param epoch: 训练迭代轮数
        """
        self.train_data = trainData
        self.train_label = trainLbl
        self.test_data = testData
        self.test_label = testLbl
        self.epoch = epoch

    def training(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    # # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")

    # EM算法流程
    em = EM(train_data, train_lbl, test_data, test_lbl,
            epoch=50)

