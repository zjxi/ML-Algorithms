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
                C[int(k)] += 1
    # 计算训练集的样本总数|D|
    D = len(attrs)

    # 最后计算H(D)
    H_D = 0
    for k in range(K):
        H_D += C[k] / D * np.log2(C[k] / D)
    H_D = -H_D

    # (2) 计算特征A对训练集D的经验条件熵H(D|A)
    # 由于该训练集D有8个特征，即A1-A8
    ss_list = []
    ss = set()
    for j in range(len(A)):
        for attr in attrs:
            ss.add(attr[j])
        ss_list.append([e for e in ss])
        ss.clear()

    g_DA = [0] * len(A)
    for j, se in enumerate(ss_list):
        cls = [0] * D  # 每个特征下的不同结点数(可能值)
        lbls = np.zeros((D, K))  # 每个特征下的不同类别数
        for attr, lbl in zip(attrs, labels):
            for i in range(len(se)):
                if se[i] == attr[j]:
                    cls[i] += 1
                    lbls[i][int(lbl)] += 1

        H_DA = calc_empirical_conditional_entropy(cls, lbls, K, D)
        # (3) 计算信息增益
        g_DA[j] = H_D - H_DA

    # 返回信息增益最大的特征列标，即Aj，以及最大信息增益值
    return g_DA.index(max(g_DA)), max(g_DA)


def calc_empirical_conditional_entropy(cls, lbls, K, D):
    H_DA, H_Di = 0, 0
    for i in range(len(cls)):
        if cls[i] > 0:
            for j in range(K):
                if lbls[i][j] > 0:
                    H_Di += lbls[i][j] / cls[i] * np.log2(lbls[i][j] / cls[i])
            H_Di *= - (cls[i] / D)
            H_DA += H_Di

    return H_DA


if __name__ == '__main__':
    # 分别读取训练集、测试集
    train_attr, train_lbl = load_data("../data/train.txt")
    test_attr, test_lbl = load_data("../data/test.txt")
    # 计算最大信息增益
    print(calc_information_gain(train_attr, train_lbl, train_attr[0]))
