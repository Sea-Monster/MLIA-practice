# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


def load_data_set():
    data_mat = []
    label_mat = []
    file_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'testSet.txt')
    with open(file_full_path) as fr:
        for line in fr.readlines():
            line_array = line.strip().split()
            # TODO 第一个元素1.0 是什么意思？(就是为了输入一个不变的常量？)
            data_mat.append([1.0, float(line_array[0]), float(line_array[1])])
            label_mat.append(int(line_array[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + np.exp(-in_x))


def gradient_ascent(data_matrix_in, class_labels):
    """
    梯度上升算法
    :param data_matrix_in:
    :param class_labels:
    :return:
    """
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels).transpose()  # transpose 置换，行列互换
    m, n = np.shape(data_matrix)
    alpha = 0.001  # 向量目标移动的步长
    max_cycles = 500
    weights = np.ones((n, 1))

    # 执行500次的梯度上升
    for k in range(max_cycles):
        # data_matrix的shape为 m*n, weights的shape为n*1, 相乘后shape 为 m * 1, m行1列即是列向量
        h = sigmoid(data_matrix * weights)  # h将会是一个列向量
        error = (label_matrix - h)

        # 梯度上升：
        # data_matrix的shape为m，n。转置后为n,m。error的shape为 m * 1
        # 则alpha * alpha * data_matrix的转置 * error，shape为n*1
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    """
    画出数据集和逻辑回归最佳拟合曲线
    :param weights: 回归系数（列）向量
    :return:
    """
    data_arr, label_arr = load_data_set()
    data_array = np.array(data_arr)
    n = np.shape(data_array)[0]  # 行数
    x_cord1 = []
    y_cord1 = []

    x_cord2 = []
    y_cord2 = []

    for i in range(n):
        if int(label_arr[i]) == 1:
            x_cord1.append(data_array[i,1])
            y_cord1.append(data_array[i,2])
        else:
            x_cord2.append(data_array[i, 1])
            y_cord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1,y_cord1, s=30, c='red',marker='s')
    ax.scatter(x_cord2,y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 最佳拟合曲线 w0 + w1*x + w2*y = 0 --> y = (-w0 - w1*x) /w2
    # 这里设置sigmoid函数为0？ 0 是两个类别（类别1和类别0）的分界处
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_gradient_ascent(data_array, class_labels):
    """
    随机梯度上升算法
    :param data_array:  训练集数据
    :param class_labels:
    :return:
    """
    m, n = np.shape(data_array) #m：训练集有多少条数据，n：训练集有多少个特征
    alpha = 0.01
    # 创建一个一维数组，元素个数为n，每个元素的值为1，这个一维数组如果转化为矩阵，将会是1行n列
    weights = np.ones(n)
    for i in range(m):
        # data_arr[i]为行向量，shape为1*n
        what_mean = data_array[i] * weights

        # print('只是对应元素的乘积：')
        # print(what_mean)

        # 原来是想求点乘，然后作为sigmoid的入参。。。
        # 输入的data_array是一组X0,X1,X2...的数据，每个元素分别乘以回归系数后，再相加
        h = sigmoid(sum(data_array[i] * weights))

        # 还不如这样写
        h = sigmoid(np.dot(data_array[i], weights))

        error = class_labels[i] - h
        weights = weights + alpha * error * data_array[i]
    return weights


def stoc_gradient_ascent_advance(data_array, class_labels, num_iter=150):
    """
    改进的随机梯度上升算法
    :param data_array:
    :param class_labels:
    :param num_iter:
    :return:
    """
    # m：训练集有多少条数据，n：训练集有多少个特征
    m, n = np.shape(data_array)
    # 创建一个一维数组，元素个数为n，每个元素的值为1，这个一维数组如果转化为矩阵，将会是1行n列
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # alpha每次迭代需要调整
            # TODO 为什么要这样调整？
            alpha = 4/(1.0+j+i)+0.01

            # 随机选取更新
            random_index = int(np.random.uniform(0, len(data_index)))

            # 求点乘，然后作为sigmoid的入参。。。
            # 输入的data_array是一组X0,X1,X2...的数据，每个元素分别乘以回归系数后，再相加
            h = sigmoid(np.dot(data_array[random_index], weights))

            error = class_labels[random_index] - h
            weights = weights + alpha * error * data_array[random_index]
            del(data_index[random_index])
    return weights


if __name__ == '__main__':
    # data_matrix, label_matrix = load_data_set()
    # print(data_matrix)
    # print('\n\n')
    # print(label_matrix)


    # data_array, label_matrix = load_data_set()
    # weights = gradient_ascent(data_array, label_matrix)
    # plot_best_fit(weights)

    # data_array, label_matrix = load_data_set()
    # weights = stoc_gradient_ascent(np.array(data_array), label_matrix)
    # print(weights)
    # plot_best_fit(weights)

    data_array, label_matrix = load_data_set()
    weights = stoc_gradient_ascent_advance(np.array(data_array), label_matrix)
    print(weights)
    plot_best_fit(weights)

