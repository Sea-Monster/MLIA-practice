# -*- coding: utf-8 -*-
from math import log
import operator


def calc_shannon_ent(data_set):
    """
    求数据集的香农熵
    :param data_set: 估计是这种格式，每一行是一个list，list最后一个元素就是标签，其他元素是特征
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}

    # 为所有可能分类创建字典
    for feature_vec in data_set:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0

    # 求香农熵
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # 该标签在数据集中出现的概率
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    按照给定特征划分数据集，筛选某个特征为指定特征值的数据
    （然后因为是按该特征进行划分了，该特征在以后的划分中就不用再出现，所以把该特征在新的列表中移除）
    :param data_set:    待划分的数据集，格式如下，每一行是一个list，list最后一个元素就是标签，其他元素是特征
    :param axis:        划分数据集的特征(特征的序号)
    :param value:       需要返回的特征的值(筛选特征的值要等于此值)
    :return:
    >>>myDat = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    >>>split_data_set(myDat,0,1)
    [[1, 'yes'], [1, 'yes'], [0, 'no']]
    >>>split_data_set(myDat,0,0)
    [[1, 'no'], [1, 'no']]

    """
    # 创建新的list对象
    ret_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            # 抽取, 把指定特征从列表中去掉，组成一个新的特征+标签的列表
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis + 1:])
            ret_data_set.append(reduced_feature_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    # 特征数，取第一个样本即可得知，由于最后一个元素是标签，所以特征数是长度-1
    num_features = len(data_set[0]) - 1

    # 计算整个数据集的熵（无序度）
    base_entropy = calc_shannon_ent(data_set)

    best_info_gain = 0.0
    best_feature = -1

    # 遍历数据集的特征1，组成一个新的数组1， 遍历数据集的特征2，组成一个新的数组2...
    # 我的理解是，收集每一个特征都会有哪些特征值
    for i in range(num_features):
        # 创建唯一的分类标签列表
        feature_list = [example[i] for example in data_set]

        # 每一组特征值列表中，去掉重复的特征值
        unique_vals = set(feature_list)
        new_entropy = 0.0

        # 计算每种划分方式的信息熵
        for value in unique_vals:
            # 原数据集剔除了某个特征值之后的数据集
            sub_data_set = split_data_set(data_set, i, value)

            # 该特征值在数据集中出现的概率
            prob = len(sub_data_set) / float(len(data_set))

            # 计算划分后的子数据集的熵值（信息期望值总和）
            new_entropy += prob * calc_shannon_ent(sub_data_set)

        # 整个数据集的熵，减去划分后的子数据集的熵，得出的是信息增益？这是什么东西呢？
        # 为什么是减？-- 信息增益是熵的减少或者是数据无序度的减少
        info_gain = base_entropy - new_entropy

        if (info_gain > best_info_gain):
            # 计算最好的信息增益
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    从标签列表中得出出现次数最多的标签
    :param class_list: 应该是标签的列表
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set:    数据集，应该是一个由多个[特征值1，特征值2...., 分类标签]组成的二维数组
    :param labels:      标签列表，包含了数据集中所有特征的标签，此算法本身其实不需要此变量
    :return:
    >>>data_set = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    >>>labels = ['能够不浮出水面', '有鳍']
    >>>myTree = create_tree(data_set, labels)
    >>>myTree
    {'能够不浮出水面':{0:'no', 1:{'有鳍':{0:'no', 1:'yes'}}}}
    """
    # data_set中每个元素中的最后一个是分类标签，把它们全部提取出来，组成分类标签的列表
    class_list = [example[-1] for example in data_set]

    # 类别完全相同则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 遍历完所有特征时返回出现次数最多的
    if len(data_set[0]) == 1:  # 特征都分类完了，只剩下分类标签了，所以数组大小为1
        return majority_cnt(class_list)

    # 特征的序号
    best_feature = choose_best_feature_to_split(data_set)
    # 特征的名字（只为了给出数据明确的含义，显示用）
    best_feature_label = labels[best_feature]  # 特征的名字

    my_tree = {best_feature_label: {}}

    # 得到列表包含的所有属性值
    del (labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]  # 特征值列表
    unique_vals = set(feature_values)  # 特征值列表去重
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(
            split_data_set(data_set, best_feature, value),
            sub_labels
        )

    return my_tree