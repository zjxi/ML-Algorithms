"""
    Implementation of the ID3 decision tree,
    written by zjxi, 2020/11/1
"""
import numpy as np


def load_data(filename):
    """
    导入训练集或者测试集数据
    :param filename:
    :return:
    """
    attr, label = [], []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split(',')
            # 添加标签向量
            label.append(line[-1])
            # 添加特征向量
            attr.append(line[:-1])

    return attr, label


def calc_empirical_entropy(labels):
    """
    计算经验熵
    :param labels: 训练集的特征向量集合
    :return: 训练集D的经验熵
    """
    s = set([lbl for lbl in labels])
    K = len(s)
    # 计算每个类别的|Ck|
    C = [0] * len(s)
    for lbl in labels:
        for k in s:
            if lbl == k:
                C[int(k)] += 1
    # 计算训练集的样本总数|D|
    D = len(labels)
    # 最后根据公式5.7，计算H(D)
    H_D = 0
    for k in range(K):
        H_D += C[k] / D * np.log2(C[k] / D)

    return -H_D


def calc_empirical_conditional_entropy(prob, cls, K, D):
    """
    计算经验条件熵
    :param cls: 不同可能值结点对应数量的数组
    :param lbls: 不同类别数量的数组
    :param K: 全部类别数
    :param D: 训练数据集的样本总量
    :return: 不同特征A对于训练集D的经验条件熵
    """
    # 根据公式5.8
    # H_DA为特征Aj对训练集D的经验条件熵，H_Di为子集Di的属于不同类别的集合
    H_DA, H_Di = 0, 0
    for i in range(len(cls)):
        if prob[i] > 0:
            for j in range(K):
                if cls[i][j] > 0:
                    H_Di += cls[i][j] / prob[i] * np.log2(cls[i][j] / prob[i])
            H_Di *= - (prob[i] / D)
            H_DA += H_Di
        else:
            break

    return H_DA


def calc_information_gain(attrs, labels, A):
    """
    信息增益算法流程
    :param attrs: 训练集的特征向量集合
    :param labels: 训练集的标签向量集合
    :param A: 特征向量
    :return: 最大信息增益的特征列下标、最大信息增益值
    """
    # 根据算法5.1 信息增益的算法
    # (1) 计算数据集D，即训练集的经验熵H(D)
    # 首先应该得到Ck对应每个类别的样本数|Ck|，训练集D的样本总数|D|
    # K为类别总数
    K = len(set([lbl for lbl in labels]))
    # 计算经验熵
    H_D = calc_empirical_entropy(labels)

    # (2) 计算特征A对训练集D的经验条件熵H(D|A)
    # 由于该训练集D有8个特征，即A1-A8
    D = len(labels)
    ss_list = []
    for j in range(len(A)):
        ss = set(attr[j] for attr in attrs)
        ss_list.append([e for e in ss])

    g_DA = [0] * len(A)
    for j, se in enumerate(ss_list):
        prob = [0] * D  # 每个特征下的相同结点(可能值)的总数
        cls = np.zeros((D, K))  # 相同结点下的不同类别数
        for attr, lbl in zip(attrs, labels):
            for i in range(len(se)):
                if se[i] == attr[j]:
                    prob[i] += 1
                    cls[i][int(lbl)] += 1
        # 计算经验条件熵
        H_DA = calc_empirical_conditional_entropy(prob, cls, K, D)
        # (3) 计算信息增益
        # 根据公式5.9
        g_DA[j] = H_D - H_DA

    # 返回信息增益最大的特征列标，即Aj，以及最大信息增益值
    return g_DA.index(max(g_DA)), max(g_DA)


if __name__ == '__main__':
    # 分别读取训练集、测试集
    train_attr, train_lbl = load_data("../data/train.txt")
    test_attr, test_lbl = load_data("../data/test.txt")
    # 计算最大信息增益
    print(calc_information_gain(train_attr, train_lbl, train_attr[0]))
