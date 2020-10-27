"""
    Implementation of the naive bayers,
    written by zjxi, 2020/10/26
"""
import numpy as np


def load_data(file_name):
    """
    从给定的路径读取数据集
    :param file_name: 数据集文件路径
    :return: 特征数据集，标签数据集
    """
    print("正在读取数据...")
    # 分别创建特征数组、标签数组
    attrs, labels = [], []
    # 读取数据集
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            line = line.replace("\n", "").split(',')
            # 添加标签向量
            labels.append(line[-1])
            # 添加特征向量
            attrs.append([x for x in line[0:9]])

    return attrs, labels


class NaiveBayers:
    def __init__(self, trainAttr, trainLbl, testAttr, testLbl):
        # 声明训练、测试特征集和标签集
        self.train_attr = trainAttr
        self.test_attr = testAttr
        self.train_lbl = trainLbl
        self.train_lbl = testLbl
        # 声明先验概率、条件概率
        self.P_y = None
        self.P_x_y = None

    def calculate_priori_prob(self):
        """
        计算先验概率
        :param attrs: 特征向量
        :param labels: 标签向量
        :return:
        """
        # 计算训练集中类别的种类数
        s = set()
        for lbl in self.train_lbl:
            s.add(lbl)
        class_num = len(s)
        # 定义类别数组，并将类别作为下标(这里规定类别均为0-n的整数)并进行相应的标签总数统计
        classes = [0] * class_num
        for lbl in self.train_lbl:
            for cls in s:
                if lbl == cls:
                    classes[int(cls)] += 1
        # 计算不同标签的先验概率
        P_y = [0] * class_num
        lambd = 1
        for i in range(len(classes)):
            # 根据4.2.3部分的贝叶斯估计，避免因极大似然估计存在估计概率为0的情况，
            # 因此需要在分子处加上lambda>=0，在分母上加上K*lambda(K一般取标签的种类总数)
            P_y[i] = (classes[i] + lambd) / (len(labels) + class_num * lambd)
        # 对先验概率取对数，由于计算结果是一个很小的接近0的数，在程序运行中很可能会向下溢现象无法比较。
        # 因此进行log处理，log在定义域内是一个递增函数，也就是说log（x）中，x越大，log也就越大，单调性和原数据保持一致。
        # 所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。
        self.P_y = np.log(P_y)

    def calculate_conditional_prob(self):
        pass


if __name__ == '__main__':
    nb = NaiveBayers()
    attrs, labels = nb.load_data("../data/train.txt")
    nb.calculate_priori_prob(attrs, labels)
