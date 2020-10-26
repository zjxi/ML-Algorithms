"""
    Implementation of the perceptron with duality,
    written by zjxi, 2020/10/23
"""
import numpy as np


def load_data(file_name):
    print("正在读取数据...")
    # 分别创建特征数组、标签数组
    features, labels = [], []
    # 读取数据集
    with open(file_name, 'r') as fr:
        for line in fr.readlines():
            # 添加标签
            line = line.replace("\n", "").split(',')
            lbl = line[-1]
            if int(lbl) == 0:
                labels.append(-1)
            else:
                labels.append(1)
            # 添加特征向量
            features.append([float(num) for num in line[0:9]])

    return features, labels


class PerceptronDual:
    def __init__(self, trainData, trainLbl, testData, testLbl, lr=0.0001, epoch=50):
        self.train_data = trainData
        self.train_label = trainLbl
        self.test_data = testData
        self.test_label = testLbl
        self.epoch = epoch
        # 初始化权重，偏置项，学习率
        print(np.shape(self.train_data))
        self.a = 0
        self.b = 0
        self.lr = lr
        # 分别将特征向量、标签向量转化为矩阵并进行相应的转置变化
        self.train_data = np.mat(self.train_data)
        self.train_label = np.mat(self.train_label).T
        self.test_data = np.mat(self.test_data)
        self.test_label = np.mat(self.test_label).T

    def train(self):
        print("开始训练模型...")

        m, n = np.shape(self.train_data)
        # 迭代轮数
        for k in range(self.epoch):
            print("当前训练轮数为: %d/%d" % (k, self.epoch))
            # 在训练集中选取数据(xi, yi);
            for i in range(m):
                # 获取当前行的特征向量
                xi = self.train_data[i]
                # 获取当前行的标签
                yi = self.train_label[i]
                # 如果当前存在分类误差
                if yi * (self.a * xi.T * yi + self.b) <= 0:
                    self.a += self.lr * yi * xi
                    self.b += self.lr * yi

        print("训练阶段结束！")

    def test(self):
        print("开始测试...")
        m, n = np.shape(self.test_data)

        # 分类错误的样本数
        error_rate = 0.0
        for i in range(m):
            xi = self.test_data[i]
            yi = self.test_label[i]
            if yi * (self.a * xi.T * yi + self.b) <= 0:
                error_rate += 1

        return 1 - error_rate / m


if __name__ == '__main__':
    # 读取训练数据集
    train_data, train_lbl = load_data("train.txt")
    # 读取测试数据集
    test_data, test_lbl   = load_data("test.txt")
    # 感知机算法
    p = PerceptronDual(train_data, train_lbl, test_data, test_lbl,
                       lr=0.0001, epoch=75)
    # 训练模型
    p.train()
    # 测试模型
    acc = p.test()

    print("分类准确率为：%f" % acc)
