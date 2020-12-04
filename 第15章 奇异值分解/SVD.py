"""
    Implementation of the Singular Value Decomposition
    written by zjxi @ 2020/11/30
"""

import numpy as np


def load_data(filename):
    """
    导入训练集或者测试集数据
    :param filename: 数据文件路径
    :return: 特征向量，标签向量
    """
    feat = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split(',')
            # 添加特征向量
            feat.append([float(e) for e in line[:-1]])

    return feat


# 读取数据，使用自己数据集的路径。
train_data = load_data("../data/train.txt")
train_data = np.mat(train_data)
print(train_data.shape)


# 数据必需先转为浮点型，否则在计算的过程中会溢出，导致结果不准确
train_dataFloat = train_data / 255.0
# 计算特征值和特征向量
eval_sigma1, evec_u = np.linalg.eigh(train_dataFloat.dot(train_dataFloat.T))


# 降序排列后，逆序输出
eval1_sort_idx = np.argsort(eval_sigma1)[::-1]
# 将特征值对应的特征向量也对应排好序
eval_sigma1 = np.sort(eval_sigma1)[::-1]
evec_u = evec_u[:,eval1_sort_idx]
# 计算奇异值矩阵的逆
eval_sigma1 = np.sqrt(eval_sigma1)
eval_sigma1_inv = np.linalg.inv(np.diag(eval_sigma1))
# 计算右奇异矩阵
evec_part_v = eval_sigma1_inv.dot((evec_u.T).dot(train_dataFloat))

