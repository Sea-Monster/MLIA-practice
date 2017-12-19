# -*- coding: utf-8 -*-
import numpy as np


def load_data_set():
    """
    创建实验样本
    :return:    第一个变量是进行词条切分后的文档集合，第二个变量是类别标签的集合
    """
    posting_list = [[' my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', ' to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1代表侮辱性位子，0代表正常言论
    class_vector = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vector


def create_vocabulary_list(data_set):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param data_set:
    :return:
    """
    # 创建一个空集
    vocabulary_set = set([])
    for document in data_set:
        # 创建两个集合的并集
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)


def set_of_words_2_vector(vocabulary_list, input_set):
    """

    :param vocabulary_list:     词汇的列表
    :param input_set:           某个文档
    :return:                    文档向量，向量的每一个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
    """
    # 创建一个其中所含元素都为0的向量
    return_vector = [0] * len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print('单词：%s 不在词汇表中！' % word)
    return return_vector


def train_NB0(train_matrix, train_category):
    """

    :param train_matrix:    训练文档矩阵（准确来说只是python原生二维数组）
    :param train_category:  训练文档矩阵对应的分类（一维向量）
    :return: (在非侮辱性文档类别下词汇表中单词的出现概率向量, 在侮辱性文档类别下词汇表中单词的出现概率向量, 任意文档属于侮辱性文档的概率)
    """
    # 训练文档的数目
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    # 分类为"侮辱性文档"的概率: 分类为"侮辱性文档"的文档数目，除以训练文档的总数目
    # train_category 为一维向量，只有0和1两种值，其中1代表侮辱性
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率：
    # p0_num: 对于分类0，词汇表中单词的出现次数
    # p1_num: 对于分类1，词汇表中单词的出现次数
    p0_num = np.zeros(num_words)
    p1_num = np.zeros(num_words)

    # 这两个是充当分母的？
    p0_denom = 0.0
    p1_denom = 0.0

    for i in range(num_train_docs):
        if train_category[i] == 1:  # 类别为"侮辱"
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])  # 训练文档中的单词同时也在词汇表中存在的数目
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])  # 训练文档中的单词同时也在词汇表中存在的数目

    # 概率向量
    # 分母是，对于分类1（侮辱性文档），一共出现过多少个单词（所有文档中）
    # 分子是，对于分类1（侮辱性文档），词汇表中每个单词各自出现的次数（所有文档中）
    # 相除后，得到一个对于分类1（侮辱性文档），词汇表中每个单词出现的概率
    p1_vector = p1_num / p1_denom

    # 对每个元素做除法
    p0_vector = p0_num / p0_denom

    return p0_vector, p1_vector, p_abusive


def train_NB1(train_matrix, train_category):
    """
    train_NB0函数的改进
    :param train_matrix:    训练文档矩阵（准确来说只是python原生二维数组）
    :param train_category:  训练文档矩阵对应的分类（一维向量）
    :return: (在非侮辱性文档类别下词汇表中单词的出现概率向量, 在侮辱性文档类别下词汇表中单词的出现概率向量, 任意文档属于侮辱性文档的概率)
    """
    # 训练文档的数目
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    # 分类为"侮辱性文档"的概率: 分类为"侮辱性文档"的文档数目，除以训练文档的总数目
    # train_category 为一维向量，只有0和1两种值，其中1代表侮辱性
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率：初始化为1而不是0
    # p0_num: 对于分类0，词汇表中单词的出现次数
    # p1_num: 对于分类1，词汇表中单词的出现次数
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)

    # 这两个是充当分母，
    # TODO 为什么是2？为什么不是1或者其他数？
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:  # 类别为"侮辱"
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])  # 训练文档中的单词同时也在词汇表中存在的数目
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])  # 训练文档中的单词同时也在词汇表中存在的数目

    # 概率向量(转为对数)
    # 分母是，对于分类1（侮辱性文档），一共出现过多少个单词（所有文档中）
    # 分子是，对于分类1（侮辱性文档），词汇表中每个单词各自出现的次数（所有文档中）
    # 相除后，得到一个对于分类1（侮辱性文档），词汇表中每个单词出现的概率
    p1_vector = np.log(p1_num / p1_denom)

    # 对每个元素做除法
    p0_vector = np.log(p0_num / p0_denom)

    return p0_vector, p1_vector, p_abusive


def classify_NB(vector_2_classify, p0_vector, p1_vector, p_class_1):
    """
    朴素贝叶斯分类函数

    这里边的公式都TMD怎么来的？！

    :param vector_2_classify:   要分类的向量
    :param p0_vector:           训练集中类别为"非侮辱性"的文档中，词汇表中每个单词的出现概率（组成一个一维向量）
    :param p1_vector:           训练集中类别为"侮辱性"的文档中，词汇表中每个单词的出现概率（组成一个一维向量）
    :param p_class_1:           训练集中类别为"侮辱性"文档的出现概率
    :return:
    """
    p1 = sum(vector_2_classify * p1_vector) + np.log(p_class_1)
    p0 = sum(vector_2_classify * p0_vector) + np.log(1.0 - p_class_1)
    print('p1:')
    print(p1)
    print('p0:')
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0


def test_classify(test_entry):
    list_o_posts = [[' my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', ' to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    list_classes = [0, 1, 0, 1, 0, 1]
    my_vocab_list = create_vocabulary_list(list_o_posts)
    train_mat = []
    for post_in_doc in list_o_posts:
        train_mat.append(set_of_words_2_vector(my_vocab_list, post_in_doc))

    # p0_v, p1_v, p_ab = train_NB0(train_mat, list_classes)
    p0_v, p1_v, p_ab = train_NB1(train_mat, list_classes)

    test_mat = set_of_words_2_vector(my_vocab_list, test_entry)
    # 把转换后的测试集，再转为numpy中的array
    test_doc_array = np.array(test_mat)
    classify_type = classify_NB(test_doc_array, p0_v, p1_v, p_ab)
    print('分类为：%d' %classify_type)


if __name__ == '__main__':
    # list_o_posts = [[' my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
    #                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    #                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    #                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    #                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', ' to', 'stop', 'him'],
    #                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # list_classes = [0, 1, 0, 1, 0, 1]
    # my_vocab_list = create_vocabulary_list(list_o_posts)
    # train_mat = []
    # for post_in_doc in list_o_posts:
    #     train_mat.append(set_of_words_2_vector(my_vocab_list, post_in_doc))
    #
    # p0_v, p1_v, p_ab = train_NB0(train_mat, list_classes)

    test_classify(['love', 'my', 'dalmation'])

    print('-' * 50)

    test_classify(['stupid','garbage'])
