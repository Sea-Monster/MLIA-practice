# -*- coding: utf-8 -*-
from math import log


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
        prob = float(label_counts[key]) / num_entries   # 该标签在数据集中出现的概率
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
            reduced_feature_vec.extend(feature_vec[axis+1:])
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