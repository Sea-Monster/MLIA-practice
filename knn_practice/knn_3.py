# -*- coding: utf-8 -*-
import numpy as np
import os
from knn_practice.knn_1 import classify0


# 把32✖️32的二进制图像矩阵转换为1✖️1024的向量
def img2vector(filename):
    return_vect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    training_file_list = os.listdir(
        os.path.join(os.getcwd(), 'trainingDigits')
    )

    test_file_list = os.listdir(
        os.path.join(os.getcwd(), 'testDigits')
    )

    hw_labels = []
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        files_str = filename_str.split('.')[0]
        # 文件名已经标明了该图像代表的标签（数字）
        class_num_str = int(files_str.split('_')[0])
        hw_labels.append(class_num_str)
        full_file_path = os.path.join(
            os.getcwd(), 'trainingDigits', filename_str)
        training_mat[i] = img2vector(full_file_path)

    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename_str = test_file_list[i]
        files_str = filename_str.split('.')[0]
        # 文件名已经标明了该图像代表的标签（数字）
        class_num_str = int(files_str.split('_')[0])
        full_file_path = os.path.join(
            os.getcwd(), 'testDigits', filename_str)
        vector_under_test = img2vector(full_file_path)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print('分类器判断该数字为：%d, 实际的数字为：%d' % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
        print('错误总数：%d' % error_count)
        print('错误率：%f' % (error_count / float(m_test)))


if __name__ == '__main__':
    handwriting_class_test()