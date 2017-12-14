# -*- coding: utf-8 -*-
import numpy as np
import os
from knn_practice.knn_1 import classify0


def file2matrix(filename):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    with open(filename) as fr:
        arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)  # 文件行数

    # 创建返回的Numpy矩阵
    # 文件行数就是样本数
    # 每个样本有3种特征：每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每周消费的冰淇淋公升数
    return_mat = np.zeros((numberOfLines, 3))

    class_label_vector = []
    index = 0

    # 解析文件数据到列表
    for line in arrayOlines:
        line = line.strip()  # 去除每一行前后的空格
        list_from_line = line.split('\t')
        return_mat[index] = list_from_line[0:3]
        # 逗号以及后边的冒号应该可以省略
        # return_mat[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def autoNorm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def classify_person():
    result_list = ['不喜欢', '有点喜欢','很喜欢']
    percent_tats = float(input("玩视频游戏所消耗时间百分比:\n"))
    ff_miles = float(input('每年获取的飞行常客里程数:\n'))
    ice_cream = float(input('每周消费的冰淇淋公升数:\n'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classify_result = classify0((in_arr-min_vals)/ranges, norm_mat, dating_labels, 3)
    print('你会喜欢这个人的程度：', result_list[classify_result - 1])



if __name__ == '__main__':
    classify_person()