"""
    Implementation of the K Means Clustering,
    written by zjxi @ 2020/11/21
"""

from numpy import *
import time
import matplotlib.pyplot as plt


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

# 计算欧式距离
def euclDistance(vector1, vector2):
    return sqrt(sum(pow(vector2 - vector1, 2)))  # pow()是自带函数


# 使用随机样例初始化质心
def initCentroids(dataSet, k):
    # k是指用户设定的k个种子点
    # dataSet - 此处为mat对象
    numSamples, dim = dataSet.shape
    # numSample - 行，此处代表数据集数量  dim - 列，此处代表维度，例如只有xy轴的，dim=2
    centroids = zeros((k, dim))  # 产生k行，dim列零矩阵
    for i in range(k):
        index = int(random.uniform(0, numSamples))  # 给出一个服从均匀分布的在0~numSamples之间的整数
        centroids[i, :] = dataSet[index, :]  # 第index行作为种子点（质心）
    return centroids


# k均值聚类
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # frist column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0  # 最小距离
            minIndex = 0  # 最小距离对应的点群
            ## for each centroid
            ## step2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])  # 计算到数据的欧式距离
                if distance < minDist:  # 如果距离小于当前最小距离
                    minDist = distance  # 则最小距离更新
                    minIndex = j  # 对应的点群也会更新

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:  # 如当前数据不属于该点群
                clusterChanged = True  # 聚类操作需要继续
                clusterAssment[i, :] = minIndex, minDist ** 2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  # 取列
            # nonzeros返回的是矩阵中非零的元素的[行号]和[列号]
            # .A是将mat对象转为array
            # 将所有等于当前点群j的，赋给pointsInCluster，之后计算该点群新的中心
            centroids[j, :] = mean(pointsInCluster, axis=0)  # 最后结果为两列，每一列为对应维的算术平方值

    print
    "Congratulations, cluster complete!"
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape  # numSample - 样例数量  dim - 数据的维度
    if dim != 2:
        print
        "Sorry! I can not draw because the dimension os your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print
        "Sorry! Your k is too large! Please contact Zouxy"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], ms=12.0)
    plt.show()


if __name__ == '__main__':
    # 导入数据集
    data = load_data("../data/train.txt")
    # k均值聚类
    data = array(data)
    k = 4
    centroids, clusterAssment = kmeans(data, k)

    # 聚类结果展示
    showCluster(data, k, centroids, clusterAssment)