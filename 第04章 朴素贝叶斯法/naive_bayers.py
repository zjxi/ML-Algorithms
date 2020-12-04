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
            attrs.append([x for x in line[0:8]])

    return attrs, labels


class NaiveBayers:
    def __init__(self, trainAttr, trainLbl, testAttr, testLbl):
        # 声明训练、测试特征集和标签集
        self.train_attr = trainAttr
        self.test_attr = testAttr
        self.train_lbl = trainLbl
        self.train_lbl = testLbl
        # 声明先验概率、条件概率
        self.P_y = []
        self.P_x_y = {}
        # 类别数
        self.classes = []

    def calculate_priori_conditional_prob(self):
        """
        计算先验概率、条件概率
        """
        print("正在计算先验概率、条件概率...")
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
        self.classes = classes
        # 计算不同标签的先验概率
        P_y = [0] * class_num
        lambd = 1
        for i in range(len(classes)):
            # 根据4.2.3部分的贝叶斯估计，避免因极大似然估计存在估计概率为0的情况，
            # 因此需要在分子处加上lambda>=0，在分母上加上K*lambda(K一般取标签的种类总数)
            P_y[i] = (classes[i] + lambd) / (len(self.train_lbl) + class_num * lambd)

        # 对先验概率取对数，由于计算结果是一个很小的接近0的数，在程序运行中很可能会向下溢现象无法比较。
        # 因此进行log处理，log在定义域内是一个递增函数，也就是说log（x）中，x越大，log也就越大，单调性和原数据保持一致。
        # 所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。
        self.P_y = np.log(P_y)

        # 计算不同标签下的每个特征的计数
        # 为了使该算法能够适用于不同的数据集类型，如整型、浮点型、文本型等，故采用字典的形式
        # 并将每个单独特征值和对应的标签组合成一个特定的字符串形式，如"attr*lbl"，作为key值
        P_x_y = {}
        for attrs, lbl in zip(self.train_attr, self.train_lbl):
            # 将每个"特征列下标+特征值+标签"的key进行赋值以便于后续进行自增运算
            for i, attr in enumerate(attrs):
                key = str(i) + '*' + attr + '*' + lbl
                P_x_y[key] = 0
        # 对每个特征值+标签的key下不同条件进行计数
        for attrs, lbl in zip(self.train_attr, self.train_lbl):
            for i, attr in enumerate(attrs):
                key = str(i) + '*' + attr + '*' + lbl
                P_x_y[key] += 1

        # 条件概率
        for k, v in P_x_y.items():
            for i in range(len(classes)):
                # 当是同一类别时，进行相应条件概率计算
                if str(k).split('*')[-1] == str(i):
                    P_x_y[k] = np.log((v + lambd) / (classes[i] + class_num * lambd))
        self.P_x_y = P_x_y

    def classifier(self, pred_attr):
        """
        分类器，进行预测
        :return:
        """
        # 根据输入的特征向量进行概率估计
        P = [0] * len(self.classes)
        for cls in range(len(self.classes)):
            _sum = 0
            # 按照标签顺序依次进行概率估计，由于条件概率和先验概率已经经过log运算，
            # 即公式4.6的连乘可以用连加代替
            for j, attr in enumerate(pred_attr):
                key = str(j) + '*' + attr + '*' + str(cls)
                try:  # 若当前列不存在该特征值则捕捉异常，并将其条件概率赋值为0
                    _sum += self.P_x_y[key]
                except KeyError:
                    _sum += 0

            # 最后在进行先验概率的相加计算
            P[cls] = _sum + self.P_y[cls]

        # 最后返回最大估计概率，即y = arg max (P)
        return P.index(max(P))

    def test(self):
        """
        对测试集进行预测
        :return:
        """
        # 声明错误预测数
        errs = 0
        for i in range(len(self.test_attr)):
            print(f"-----正在分类预测中，已完成{i / len(self.test_attr)}%-----")
            pred = self.classifier(self.test_attr[i])
            if pred != int(test_lbl[i]):
                errs += 1
        print("测试准确率为:", 1 - (errs / len(self.test_attr)))


if __name__ == '__main__':
    # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")
    nb = NaiveBayers(train_data, train_lbl, test_data, test_lbl)
    # 计算先验概率、条件概率
    nb.calculate_priori_conditional_prob()
    # 分类预测测试
    nb.test()
