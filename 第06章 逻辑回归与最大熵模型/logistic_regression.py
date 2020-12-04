"""
    Implementation of the logistic regression,
    written by zjxi @ 2020/11/3
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


class LogisticRegression:
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
        """
        训练过程
        :return:
        """
        m, n = np.shape(self.train_data)
        for k in range(self.epoch):
            print(f"---当前迭代轮数为:{k+1}---")
            for i in range(m):
                # 获取当前特征向量
                xi = self.train_data[i]
                # 获取当前标签向量
                yi = self.train_label[i]
                # 结合公式6.5和6.6，可以得到以下过程
                # P(Y=1|x) = P_y1_x, P(Y=0|x) = P_y0_x
                P_y1_x = np.exp(self.w * xi.T) / (1 + np.exp(self.w * xi.T))
                P_y0_x = 1 / (1 + np.exp(self.w * xi.T))
                # 当分类错误时，根据6.1.3-模型参数估计部分，进行对数似然估计
                # 并且进行最大似然估计值，即通过梯度下降法
                if P_y1_x - P_y0_x >= 0 and yi != 1 or P_y1_x - P_y0_x <= 0 and yi != 0:
                    # w = w + α Σ (h(w)-yi)xi, 其中h(w) = P(Y=1|x)
                    # 通过观察h(w)可以由P(Y=1|x)等价替换，即P_y1_x
                    self.w = self.w - self.lr * (P_y1_x - yi) * xi

        print("训练结束！")

    def test(self):
        """
        测试过程
        :return:
        """
        # 声明分类错误次数
        errs = 0
        # 获取测试集的行列值
        m, n = np.shape(self.test_data)
        for i in range(m):
            # 获取测试集的特征向量
            xi = self.test_data[i]
            # 获取测试集的标签向量
            yi = self.test_label[i]
            # 这里的w权重值已经通过逻辑回归算法训练迭代得到
            P_y1_x = np.exp(self.w * xi.T) / (1 + np.exp(self.w * xi.T))
            P_y0_x = 1 / (1 + np.exp(self.w * xi.T))
            if P_y1_x > P_y0_x and yi != 1 or P_y1_x < P_y0_x and yi != 0:
                errs += 1

        # 计算该算法的测试准确率
        acc = 1 - errs / m
        print(f"该算法的测试准确率为:{acc}")


if __name__ == '__main__':
    # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")
    # 二项逻辑回归
    LR = LogisticRegression(train_data, train_lbl, test_data, test_lbl,
                            epoch=50, lr=0.0001)
    # 训练，测试分类
    LR.training()
    LR.test()

