# -*- coding: utf-8 -*-
import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    """
    分类器
    :param inX:         用于分类的输入向量(就是未分类但最终要分类的向量)
    :param dataSet:     输入的训练样本数据集
    :param labels:      标签向量
    :param k:           选择最近邻居的数目
    :return:
    """
    dataSetSize = dataSet.shape[0]

    # 计算两个向量点之间的欧氏距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 选择距离最小的k个点，确定这些元素所在的主要分类
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序
    # 将classCount字典分解为元组列表，然后使用itemgetter方法，按照第二个元素的次序（？）对元组进行排序
    # 此处的排序为逆序，即按照从最大到最小次序排序，最后返回发生频率最高的元素标签
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    result = classify0([0,0], group, labels, 3)
