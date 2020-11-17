"""
    Implementation of the K nearest neighbor,
    written by zjxi, 2020/10/24
"""

import numpy as np


def loadData(filePath):  # 读文件
    with open(filePath, 'r+') as fr:
        # with语句会自动调用close()方法，且比显式调用更安全
        lines = fr.readlines()
        data = []
        for line in lines:  # 逐行读入
            items = line.strip().split(",")
            data.append([int(items[i]) for i in range(len(items))])
    return np.asarray(data)  # 以np.ndarray类型数组返回


class kdNode:
    # 分支结点
    def __init__(self, dim, value, left, right):
        # 切割维度，切割值，左子树，右子树
        self.dim = dim
        self.value = value
        self.left = left
        self.right = right


class kdtree:
    # kd树
    """
    构建kd-tree，data_array为初始的数据集合，数据类型是np.ndarray，
    threshold是最小划分个数
    """

    def __init__(self, data_array, threshold):

        self.threshold = threshold  # 最小分支阈值，数据个数低于此值不在划分
        row, col = data_array.shape
        k = col - 1  # k指维度，即特征向量的元素个数
        """寻找方差最小的维度"""

        def getMaxDimension(data):  # data即当前待划分的数据集合
            print("当前待划分集合: ")  # 输出待分割的数据集合
            print(data)
            maxv = -1  # 记录当前最大方差
            maxi = -1  # 记录当前方差最大的维度
            for i in range(k):
                a = np.var(data[:, i])  # 计算维度i对应的方差
                print("维度" + str(i) + "的方差" + ": " + str(a))  # 输出每个维度的方差
                if a > maxv:
                    maxi = i
                    maxv = a
            return maxi, maxv  # 返回最大方差对应的维度和最大方差值

        """
        创建一个分支结点
        """
        def createNode(data):
            split_dimension, maxv = getMaxDimension(data)
            # split_dimension， maxv分别指划分轴（维度）和最大方差值
            print("划分维度:" + str(split_dimension))  # 输出划分维度
            if maxv == 0:
                # 考虑边界情况，最大方差为0时当前数据不必划分，直接作为叶子结点
                return data
            split_value = np.median(data[:, split_dimension])
            # 取当前维度下的中位数作为划分值
            print("划分值:" + str(split_value))  # 输出划分值
            maxvalue = np.max(data[:, split_dimension])  # 当前维度下的最大元素
            minvalue = np.min(data[:, split_dimension])  # 当前维度下的最小元素
            left = []  # 保存在split_dimension下小于（或等于）split_value的点
            right = []  # 保存在split_dimension下大于（或等于）split_value的点
            for i in range(len(data)):
                if split_value < maxvalue:  # 避免0，0，0，1，2这样的分不开
                    if data[i][split_dimension] <= split_value:
                        left.append(list(data[i]))
                    else:
                        right.append(list(data[i]))
                elif split_value > minvalue:  # 避免0，1，2，2，2这样的分不开
                    if data[i][split_dimension] < split_value:
                        left.append(list(data[i]))
                    else:
                        right.append(list(data[i]))
            print("left: ", end="")  # 输出左右分支集合
            print(left)
            print("right: ", end="")
            print(right)
            # 最小分支阈值，低于此值不再划分
            root = kdNode(split_dimension, split_value,
                          (createNode(np.asarray(left)) if len(left) >= threshold else np.asarray(left)),
                          (createNode(np.asarray(right)) if len(right) >= threshold else np.asarray(right)))
            # 递归建树，注意当点集中元素个数小于最小分支阈值时直接作为叶结点而不必分支
            return root

        self.root = createNode(data_array)


n = 0
"""寻找vec对应的k邻近，klist为(距离,[向量])构成的列表，存放vec的k个近邻点的信息，初始为空"""


def findn(root, vec, klist, k):
    if type(root) == np.ndarray:  # 到达叶结点
        if len(root) == 0:
            return
        temp = (root[:, :-1] - vec) ** 2
        for i in range(len(temp)):
            a = sum(temp[i])
            global n
            n += 1
            if len(klist) != k:
                klist.append((a, root[i]))
                klist.sort(key=lambda x: x[0])  # 按距离排序
            else:
                if a < klist[k - 1][0]:
                    klist[k - 1] = [a, root[i]]
                    klist.sort(key=lambda x: x[0])  # 按距离排序
    else:
        if vec[root.demo] < root.value:
            findn(root.left, vec, klist, k)
            if abs((vec[root.demo] - root.value) ** 2) < klist[len(klist) - 1][0]:
                findn(root.right, vec, klist, k)  # 回溯
        else:
            findn(root.right, vec, klist, k)
            if abs((vec[root.demo] - root.value) ** 2) < klist[len(klist) - 1][0]:
                findn(root.left, vec, klist, k)  # 回溯


""" 
选出列表中出现次数最多的元素,一个需要注意的问题是像[2,2,1,1,3]这样的怎么选，因为之前已经按距离从小到大排序，所以应选2
"""


def findMain(alist):
    hashtable = [0 for i in range(10)]
    for i in range(len(alist)):
        hashtable[alist[i]] += 1
    maxnum = -1
    main = -1
    for i in range(len(alist)):
        if hashtable[alist[i]] > maxnum:
            main = alist[i]
            maxnum = hashtable[alist[i]]
    print("预测标签：" + str(main))
    return main


"""预测给定点的标签"""


def forecast(root, data, k):
    a = []  # 作为findn方法中的klist参数
    global n
    n = 0
    findn(root, data, a, k)
    print("遍历了" + str(n) + "个点")
    L = len(a[0][1])  # 其实就是向量维度
    res = []
    for i in range(len(a)):
        res.append(a[i][1][L - 1])
    print("K邻近标签：", end="")
    print(res)
    return findMain(res)


"""用train_list建树，用KNN对test_list中的向量进行预测并输出正确率"""


def knn(train_list, test_list, k):
    root = kdtree(train_list, 10).root  # 最小划分次数设为10
    print("最小划分个数： 10000")
    print("k = " + str(k))
    num = 0
    for i in range(len(test_list)):
        a = forecast(root, np.asarray(test_list[i][:-1]), k)
        if a == test_list[i][-1]:
            num += 1
    print("正确率：" + str(num / len(test_list)))  # 预测准确率


if __name__ == '__main__':
    train_list = loadData("../data/train.txt")
    test_list = loadData("../data/test.txt")
    knn(train_list, test_list, 3)  # K值设为3
