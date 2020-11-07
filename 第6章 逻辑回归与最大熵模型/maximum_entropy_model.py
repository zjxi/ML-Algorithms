"""
    Implementation of the maximum entropy model,
    written by zjxi @ 2020/11/5
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


class MaxEntropyModel:
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

    def IIS(self):
        pass

    def training(self):
        pass

    def test(self):
        pass


def encode(featureset, label, mapping):
    encoding = []
    for (fname, fval) in featureset.items():
        if (fname, fval, label) in mapping:
            encoding.append((mapping[(fname, fval, label)], 1))
    return encoding


def calculate_empirical_fcount(train_toks, mapping):
    fcount = np.zeros(len(mapping))
    for tok, label in train_toks:
        for (index, val) in encode(tok, label, mapping):
            fcount[index] += val
    return fcount


def prob(tok, labels, mapping, weights):
    prob_dict = {}
    for label in labels:
        total = 0.0
        for (index, val) in encode(tok, label, mapping):
            total += weights[index] * val
        prob_dict[label] = np.exp(total)
    value_sum = sum(list(prob_dict.values()))
    for (label, value) in prob_dict.items():
        prob_dict[label] = prob_dict[label] / value_sum
    return prob_dict


def calculate_estimated_fcount(train_toks, mapping, labels, weights):
    fcount = np.zeros(len(mapping))
    for tok, label in train_toks:
        prob_dict = prob(tok, labels, mapping, weights)
        for label, p in prob_dict.items():
            for (index, val) in encode(tok, label, mapping):
                fcount[index] += p * val
    return fcount


def maxent_train(train_toks):
    mapping = {}  # maps (fname, fval, label) -> fid
    labels = set()
    feature_name = set()
    for (tok, label) in train_toks:
        for (fname, fval) in tok.items():
            if (fname, fval, label) not in mapping:
                mapping[(fname, fval, label)] = len(mapping)
            feature_name.add(fname)
        labels.add(label)
    C = len(feature_name) + 1
    Cinv = 1 / C
    empirical_fcount = calculate_empirical_fcount(train_toks, mapping)
    weights = np.zeros(len(empirical_fcount))

    iter = 1
    while True:
        if iter == 100:
            break
        estimated_fcount = calculate_estimated_fcount(train_toks, mapping, labels, weights)
        weights += (empirical_fcount / estimated_fcount) * Cinv
        iter += 1
    return weights, labels, mapping


if __name__ == '__main__':
    train_data = [
        (dict(a=1, b=1, c=1), '1'),
        (dict(a=1, b=1, c=0), '0'),
        (dict(a=0, b=1, c=1), '1')]

    maxent_train(train_data)




