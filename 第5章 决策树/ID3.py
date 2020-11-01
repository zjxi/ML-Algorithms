
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


def calc_information_gain(attrs, labels, A):
    # 根据算法5.1 信息增益的算法
    # (1) 计算数据集D，即训练集的经验熵H(D)
    # 首先应该得到Ck对应每个类别的样本数|Ck|，训练集D的样本总数|D|
    s = set()
    for lbl in labels:
        s.add(lbl)
    # K为类别总数
    K = len(s)

    # 计算每个类别的|Ck|
    C = [0] * K
    for lbl in labels:
        for k in s:
            if lbl == k:
                C[k] += 1
    # 计算训练集的样本总数|D|
    D = len(attrs)

    # 最后计算H(D)
    H_D = 0
    for k in K:
        H_D += C[k] / D * np.log2(C[k] / D)
    H_D = -H_D

    # (2) 计算特征A对训练集D的经验条件熵H(D|A)
    # 由于该训练集D有8个特征，即A1-A8
    Di = [0] * len(A)  # 每个类别在整个训练集中的数量
    cls = [0] * len(A)  # 每个特征下的不同类别数
    ss = set()
    for j in range(len(A)):
        for attr in attrs:
            ss.add(attr[j])
        cls[j] = len(ss)








if __name__ == '__main__':
    # 分别读取训练集、测试集
    train_attr, train_lbl = load_data("../data/train.txt")
    test_attr, test_lbl = load_data("../data/test.txt")