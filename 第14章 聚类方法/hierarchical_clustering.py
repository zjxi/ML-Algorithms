"""
    Implementation of the Hierarchical Clustering
    written by zjxi @ 2020/11/21
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


class HierarchicalClustering:
    def __init__(self, trainData):
        """
        初始化逻辑回归的各类参数
        :param trainData: 训练特征向量集
        """
        self.train_data = np.mat(trainData)

    @staticmethod
    def calc_euclidean_distance(xi, xj):
        """
        计算两个不同样本之间的欧氏距离
        :param xi: 样本xi
        :param xj: 样本xj
        :return: xi、xj的欧氏距离
        """
        return np.sqrt(np.sum(np.square(xi - xj)))

    @staticmethod
    def calc_manhattan_distance(xi, xj):
        """
        计算两个不同样本之间的曼哈顿距离
        :param xi: 样本xi
        :param xj: 样本xj
        :return: xi、xj的曼哈顿距离
        """
        return np.sum(np.abs(xi - xj))

    @staticmethod
    def calc_chebyshev_distance(xi, xj):
        """
        计算两个不同样本之间的切比雪夫距离
        :param xi: 样本xi
        :param xj: 样本xj
        :return: xi、xj的切比雪夫距离
        """
        return max(np.abs(xi - xj))

    def calc_single_linkage(self, G_p, G_q):
        pass

    def clustering(self, cls_num):
        """
        算法14.1 聚合聚类算法
        :param cls_num: 需要聚类的个数
        :return: 最终的聚类
        """
        row, col = np.shape(self.train_data)
        # (1) 计算n个样本两两之间的欧氏距离{dij}, 记作矩阵D = [dij]nxn
        # 在这里 矩阵D 等价于 D_ij
        D_ij = np.zeros((row, row))
        for i in range(row):
            for j in range(row):
                if i != j:
                    D_ij[i][j] = self.calc_euclidean_distance(self.train_data[i], self.train_data[j])

        # (2) 构造n个类, 每个类只包含一个样本
        G_i = []
        # 首先根据row个样本构建row个类, 即Gi = {xi}, i=1, 2, ... ,row
        # 为了方便计算这里只需要存储每个样本的序号即可
        # i为样本的序号，true代表当前样本没有被合并，False代表被合并为新类

        for i in range(row):
            G_i.append([i])
        # (3) 合并类间距离最小的两个类，其中最短距离为类间距离，构建一个新类
        # 声明任意两个类之间的最短距离min_dis
        # i_idx, j_idx 分别为 最短距离的样本i、样本j的下标
        min_dis = D_ij[0][1]
        i_idx, j_idx = 0, 1
        # for i in range(row):
        #     for j in range(row):
        #         if i != j:
        #             if D_ij[i][j] < min_dis:
        #                 min_dis = D_ij[i][j]
        #                 i_idx, j_idx = i, j
        for i, i_list in enumerate(G_i):
            for j, j_list in enumerate(G_i):
                if i != j:
                    if D_ij[i][j] < min_dis:
                        i_idx, j_idx = i, j

        # 合并成新类，类的下标加1
        G_i.append([i_idx, j_idx])

        # 然后将被合并的两个样本的列表分别删除
        del G_i[i_idx]
        del G_i[j_idx-1]

        # (4) 计算新类与当前各类的距离。若类的个数为1，终止计算，否则回到步(3)

        # for i in G_i[len(G_i)-1]:
        #     for k, lis in enumerate(G_i):
        #         if len(lis) == 0 and k == len(G_i)-1:
        #             continue
        #         for j in lis:
        #             if D_ij[i][j] < min_dis and i != j:
        #                 min_dis = D_ij[i][j]
        #                 i_idx, j_idx = i, j
        #
        # print(i_idx, j_idx)
        # # 合并成新类，类的下标加1
        # if i_idx or j_idx in G_i[len(G_i) - 1]:
        #     if i_idx not in G_i[len(G_i) - 1]:
        #         G_i.append(G_i[len(G_i) - 1].append(i_idx))
        #     else:
        #         G_i.append(G_i[len(G_i) - 1].append(j_idx))
        # else:
        #     G_i.append([i_idx, j_idx])
        #
        # # 然后将被合并的两个样本的列表分别清空
        # G_i[i_idx].clear()
        # G_i[j_idx].clear()

        # # 计算当前类的个数
        # cls_cur = 0
        # for i in range(len(G_i)):
        #     if len(G_i[i]) != 0:
        #         cls_cur += 1
        # # 若当前类的个数为需要聚类的个数，则终止循环，聚类结束
        # if cls_cur == cls_num:
        #     break

        # 返回最终的聚类
        return G_i


if __name__ == '__main__':
    # 读取训练数据集
    train_data = load_data("../data/train.txt")
    # 层次聚合聚类算法
    hc = HierarchicalClustering(train_data)
    cls = hc.clustering(3)
    print(cls)

