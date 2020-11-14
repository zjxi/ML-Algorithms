"""
    Implementation of the AdaBoost,
    written by zjxi @ 2020/11/9

    AdaBoost在该测试数据集上的分类准确率为：0.783582
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
            label.append(-1 if int(line[-1]) == 0 else 1)
            # 添加特征向量
            feat.append([float(e) for e in line[:-1]])

    return feat, label


def set_weakly_classifier(filename):
    _set = set()
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n').split(',')
            _set.add((line[0], line[-1]))
    print(_set)


class AdaBoost:
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
        # 初始化权重，根据算法8.1的步骤(1)
        self.W_mi = [1 / len(self.train_label)] * len(self.train_label)
        # 声明最终分类器
        self.G_x_list = []

    @staticmethod
    def weakly_classifier(idx, xi):
        """
        弱学习分类器的选择
        :param idx: 选择第idx列特征向量的分类器
        :param xi: 当前输入的特征值
        :return: 类别(-1/1)
        """
        # 根据训练集数据分布情况进行自定义弱8个学习分类器的输入阈值
        G_mx = [3.5, 130, 60, 30.5, 200, 35.5, 0.5, 40.5]
        # 根据8.1.3的例8.1的步骤(a)制定决策规则
        return -1 if xi < G_mx[idx] else 1

    def calc_classify_err(self):
        """
        计算分类误差率em
        :return: 最小的分类误差率
        """
        # 声明分类误差率
        e_m = [0] * len(self.train_data[0])
        cnt = 0
        for feat, lbl in zip(self.train_data, self.train_label):
            for i, xi in enumerate(feat):
                if lbl != self.weakly_classifier(i, xi):
                    e_m[i] += self.W_mi[cnt]
            cnt += 1
        # 计算得到并返回最小的分类误差率, 弱分类器的下标
        return e_m[int(np.argmin(e_m))], int(np.argmin(e_m))

    @staticmethod
    def calc_am_coefficient(e_m):
        """
        计算Gm(x)的系数am
        :param e_m: 最小分类误差率
        :return: Gm(x)的系数
        """
        return 1 / 2 * np.log(1 / e_m - 1)

    def calc_normalization_factor(self, G_idx, a_m):
        """
        计算规范化因子
        :param G_idx: 当前弱分类器的下标
        :param a_m: Gm(x)的系数
        :return: 规范化因子Zm
        """
        # 声明规范化因子Zm
        Z_m = 0
        cnt = 0
        for x, yi in zip(self.train_data, self.train_label):
            # 调用当前使用的弱学习分类器
            Gm_xi = self.weakly_classifier(G_idx, x[G_idx])
            # 计算规范化因子
            Z_m += self.W_mi[cnt] * np.exp(-1 * a_m * yi * Gm_xi)

        return Z_m

    def training(self):
        # 声明G(x)函数列表
        G_x_list = []
        # 更新权重分布，直到最小分类误差率em为0
        for m in range(self.epoch):
            # 获取最小分类误差率em
            e_m, G_idx = self.calc_classify_err()
            # print(e_m ,G_idx)
            # 获取Gm(x)的系数am
            a_m = self.calc_am_coefficient(e_m)
            # 计算规范化因子
            Z_m = self.calc_normalization_factor(G_idx, a_m)
            cnt = 0
            for x, yi in zip(self.train_data, self.train_label):
                # 调用当前使用的弱学习分类器
                Gm_xi = self.weakly_classifier(G_idx, x[G_idx])
                # 更新数据集的权重分布
                self.W_mi[cnt] = self.W_mi[cnt] * np.exp(-1 * a_m * yi * Gm_xi) / Z_m
                # print(self.W_mi)
                cnt += 1

            # 构建基本分类器的线性组合，根据公式(8.6)
            G_x_list.append([a_m, G_idx])
            self.G_x_list = G_x_list

            # 当最小分类误差率为0时终止迭代
            if e_m == 0:
                break

    def predict(self, feat):
        """
        利用分类器进行预测
        :param feat: 当前输入特征向量
        :return: 分类结果
        """
        # 声明最终分类器G(x)，根据公式(8.7)
        G_x = 0
        for fx in self.G_x_list:
            # 根据公式(8.6)进行计算
            G_x += fx[0] * self.weakly_classifier(fx[1], feat[fx[1]])

        return np.sign(G_x)

    def test(self):
        """
        测试集测试
        :return: 分类准确率
        """
        # 声明错误率
        errs = 0
        for feat, lbl in zip(self.test_data, self.test_label):
            if lbl != self.predict(feat):
                errs += 1
        # 返回分类器的识别准确率
        return 1 - errs / len(self.test_label)


if __name__ == '__main__':
    # 预先根据数据集设置相应数量的弱学习分类器
    # set_weakly_classifier("../data/train.txt")

    # # 读取训练数据集
    train_data, train_lbl = load_data("../data/train.txt")
    # 读取测试数据集
    test_data, test_lbl = load_data("../data/test.txt")

    # AdaBoost算法流程
    ab = AdaBoost(train_data, train_lbl, test_data, test_lbl,
                  epoch=200)
    # 训练
    ab.training()
    # 测试，并计算算法分类准确率
    acc = ab.test()
    print("AdaBoost在该测试数据集上的分类准确率为：%f" % acc)


