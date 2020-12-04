"""
    Implementation of the ID3 decision tree,
    written by zjxi, 2020/11/1
"""
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


def calc_empirical_entropy(labels):
    """
    计算经验熵
    :param labels: 训练集的标签向量集合
    :return: 训练集D的经验熵
    """
    s = set([lbl for lbl in labels])
    K = len(s)
    # 计算每个类别的|Ck|
    C = [0] * len(s)
    for lbl in labels:
        for k in s:
            if lbl == k:
                C[int(k)] += 1
    # 计算训练集的样本总数|D|
    D = len(labels)
    # 最后根据公式5.7，计算H(D)
    H_D = 0
    for k in range(K):
        H_D += C[k] / D * np.log2(C[k] / D)

    return -H_D


def calc_empirical_conditional_entropy(prob, cls, K, D):
    """
    计算经验条件熵
    :param cls: 不同可能值结点对应数量的数组
    :param lbls: 不同类别数量的数组
    :param K: 全部类别数
    :param D: 训练数据集的样本总量
    :return: 不同特征A对于训练集D的经验条件熵
    """
    # 根据公式5.8
    # H_DA为特征Aj对训练集D的经验条件熵，H_Di为子集Di的属于不同类别的集合
    H_DA, H_Di = 0, 0
    for i in range(len(cls)):
        if prob[i] > 0:
            for j in range(K):
                if cls[i][j] > 0:
                    H_Di += cls[i][j] / prob[i] * np.log2(cls[i][j] / prob[i])
            H_Di *= - (prob[i] / D)
            H_DA += H_Di
        else:
            break

    return H_DA


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
    # K为类别总数
    K = len(set([lbl for lbl in labels]))
    # 计算经验熵
    H_D = calc_empirical_entropy(labels)

    # (2) 计算特征A对训练集D的经验条件熵H(D|A)
    # 由于该训练集D有8个特征，即A1-A8
    D = len(labels)
    ss_list = []
    for j in range(len(A)):
        ss = set(attr[j] for attr in attrs)
        ss_list.append([e for e in ss])

    g_DA = [0] * len(A)
    for j, se in enumerate(ss_list):
        prob = [0] * D  # 每个特征下的相同结点(可能值)的总数
        cls = np.zeros((D, K))  # 相同结点下的不同类别数
        for attr, lbl in zip(attrs, labels):
            for i in range(len(se)):
                if se[i] == attr[j]:
                    prob[i] += 1
                    cls[i][int(lbl)] += 1
        # 计算经验条件熵
        H_DA = calc_empirical_conditional_entropy(prob, cls, K, D)
        # (3) 计算信息增益
        # 根据公式5.9
        g_DA[j] = H_D - H_DA

    # 返回信息增益最大的特征列标，即Aj，以及最大信息增益值
    return g_DA.index(max(g_DA)), max(g_DA)


def majorClass(labelArr):
    '''
    找到当前标签集中占数目最大的标签
    :param labelArr: 标签集
    :return: 最大的标签
    '''
    # 建立字典，用于不同类别的标签技术
    classDict = {}
    # 遍历所有标签
    for i in range(len(labelArr)):
        # 当第一次遇到A标签时，字典内还没有A标签，这时候直接幅值加1是错误的，
        # 所以需要判断字典中是否有该键，没有则创建，有就直接自增
        if labelArr[i] in classDict.keys():
            # 若在字典中存在该标签，则直接加1
            classDict[labelArr[i]] += 1
        else:
            # 若无该标签，设初值为1，表示出现了1次了
            classDict[labelArr[i]] = 1
    # 对字典依据值进行降序排序
    classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    # 返回最大一项的标签，即占数目最多的标签
    return classSort[0][0]


def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    '''
    更新数据集和标签集
    :param trainDataArr:要更新的数据集
    :param trainLabelArr: 要更新的标签集
    :param A: 要去除的特征索引
    :param a: 当data[A]== a时，说明该行样本时要保留的
    :return: 新的数据集和标签集
    '''
    # 返回的数据集
    retDataArr = []
    # 返回的标签集
    retLabelArr = []
    # 对当前数据的每一个样本进行遍历
    for i in range(len(trainDataArr)):
        # 如果当前样本的特征为指定特征值a
        if trainDataArr[i][A] == a:
            # 那么将该样本的第A个特征切割掉，放入返回的数据集中
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A + 1:])
            # 将该样本的标签放入返回标签集中
            retLabelArr.append(trainLabelArr[i])
    # 返回新的数据集和标签集
    return retDataArr, retLabelArr


def createTree(*dataSet):
    '''
    递归创建决策树
    :param dataSet:(trainDataList， trainLabelList) <<-- 元祖形式
    :return:新的子节点或该叶子节点的值
    '''
    # 设置Epsilon，“5.3.1 ID3算法”第4步提到需要将信息增益与阈值Epsilon比较，若小于则
    # 直接处理后返回T
    # 该值的大小在设置上并未考虑太多，观察到信息增益前期在运行中为0.3左右，所以设置了0.1
    Epsilon = 0.1
    # 从参数中获取trainDataList和trainLabelList
    # 之所以使用元祖作为参数，是由于后续递归调用时直数据集需要对某个特征进行切割，在函数递归
    # 调用上直接将切割函数的返回值放入递归调用中，而函数的返回值形式是元祖的，等看到这个函数
    # 的底部就会明白了，这样子的用处就是写程序的时候简洁一点，方便一点
    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]
    # 打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
    print('start a node', len(trainDataList[0]), len(trainLabelList))

    # 将标签放入一个字典中，当前样本有多少类，在字典中就会有多少项
    # 也相当于去重，多次出现的标签就留一次。举个例子，假如处理结束后字典的长度为1，那说明所有的样本
    # 都是同一个标签，那就可以直接返回该标签了，不需要再生成子节点了。
    classDict = {i for i in trainLabelList}
    # 如果D中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
    # 即若所有样本的标签一致，也就不需要再分化，返回标记作为该节点的值，返回后这就是一个叶子节点
    if len(classDict) == 1:
        # 因为所有样本都是一致的，在标签集中随便拿一个标签返回都行，这里用的第0个（因为你并不知道
        # 当前标签集的长度是多少，但运行中所有标签只要有长度都会有第0位。
        return trainLabelList[0]

    # 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
    # 即如果已经没有特征可以用来再分化了，就返回占大多数的类别
    if len(trainDataList[0]) == 0:
        # 返回当前标签集中占数目最大的标签
        return majorClass(trainLabelList)

    # 否则，按式5.10计算A中个特征值的信息增益，选择信息增益最大的特征Ag
    Ag, EpsilonGet = calc_information_gain(trainDataList, trainLabelList, trainDataList[0])

    # 如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck
    # 作为该节点的类，返回T
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)

    # 否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
    # 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
    treeDict = {Ag: {}}
    # 特征值为0时，进入0分支
    # getSubDataArr(trainDataList, trainLabelList, Ag, 0)：在当前数据集中切割当前feature，返回新的数据集和标签集

    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    return treeDict


def predict(testDataList, tree):
    '''
    预测标签
    :param testDataList:样本
    :param tree: 决策树
    :return: 预测结果
    '''
    # treeDict = copy.deepcopy(tree)

    # 死循环，直到找到一个有效地分类
    while True:
        # 因为有时候当前字典只有一个节点
        # 例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
        # 若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value
        (key, value), = tree.items()
        # 如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            # 获取目前所在节点的feature值，需要在样本中删除该feature
            # 因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            # 所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性
            dataVal = testDataList[key]
            del testDataList[key]
            # 将tree更新为其子节点的字典
            tree = value[dataVal]
            # 如果当前节点的子节点的值是int，就直接返回该int值
            # 例如{403: {0: 7, 1: {297:7}}，dataVal=0
            # 此时上一行tree = value[dataVal]，将tree定位到了7，而7不再是一个字典了，
            # 这里就可以直接返回7了，如果tree = value[1]，那就是一个新的子节点，需要继续遍历下去
            if type(tree).__name__ == 'int':
                # 返回该节点值，也就是分类值
                return tree
        else:
            # 如果当前value不是字典，那就返回分类值
            return value


def test(testDataList, testLabelList, tree):
    '''
    测试准确率
    :param testDataList:待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''
    # 错误次数计数
    errorCnt = 0
    # 遍历测试集中每一个测试样本
    for i in range(len(testDataList)):
        # 判断预测与标签中结果是否一致
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    # 返回准确率
    return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    # 分别读取训练集、测试集
    train_attr, train_lbl = load_data("../data/train.txt")
    test_attr, test_lbl = load_data("../data/test.txt")
    # 计算最大信息增益
    # print(calc_information_gain(train_attr, train_lbl, train_attr[0]))
    # 创建ID3决策树
    tree = createTree((train_attr, train_lbl))
    # 测试
    acc = test(test_attr, test_lbl, tree)
    print(acc)
