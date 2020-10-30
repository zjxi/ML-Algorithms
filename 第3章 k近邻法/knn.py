"""
    Implementation of the K nearest neighbor,
    written by zjxi, 2020/10/24
"""

import numpy as np


def load_data(file_name):
    print("正在读取数据...")
    # 分别创建特征数组、标签数组
    features, labels = [], []
    # 读取数据集
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            line = line.replace("\n", "").split(',')
            # 添加标签向量
            labels.append(line[-1])
            # 添加特征向量
            features.append(line[0:8])

    return features, labels

class Node:
    def __init__(self):



class KNN:
    def __init__(self, trainAttr, trainLbl, testAttr, testLbl):
        # 声明训练、测试特征集和标签集
        self.train_attr = trainAttr
        self.test_attr = testAttr
        self.train_lbl = trainLbl
        self.train_lbl = testLbl

    @staticmethod
    def calculate_distance(xi, xj):
        """
        计算两个特征向量的欧式距离
        :param xi: 特征向量xi
        :param xj: 特征向量xj
        :return:
        """
        return np.sqrt(np.sum(np.square(xi - xj)))

    def create_kd_tree(self, points, depth):
        """
        构造平衡kd树
        :return:
        """


    def search_kd_tree(self):
        """
        利用kd树的最近邻搜索
        :return:
        """
        pass

    def test(self):
        """
        利用测试数据集进行预测
        :return:
        """
        pass


if __name__ == '__main__':
    # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")
