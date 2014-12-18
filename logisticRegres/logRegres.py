#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import numpy
import random
import matplotlib.pyplot as plt


def load_data_set():
    """
    从文件中读取数据
    :return data_list:列表，每个元素也是列表，表示一个数据的所有特征值
    :return label_list:列表，每个元素是数据的类别
    """
    data_list = []
    label_list = []
    file_reader = open('testSet.txt')
    for line in file_reader.readlines():
        line_array = line.strip().split()
        data_list.append([1.0, float(line_array[0]), float(line_array[1])])
        label_list.append(int(line_array[2]))
    return data_list, label_list


def sigmoid(x):
    """
    返回输入数据输入Sigmoid函数的结果
    :param x:列表，输入数据
    :return:列表，函数的结果
    """
    return 1.0 / (1 + numpy.exp(-x))


def grad_ascent(data_list, label_list):
    """
    得到梯度上升算法中的权值w
    :param data_list:列表，每个元素也是列表，表示一个数据的所有特征值
    :param label_list:列表，每个元素是数据的类别
    :return:array，梯度上升算法的权值
    """
    data_matrix = numpy.mat(data_list)
    label_matrix = numpy.mat(label_list)
    m, n = numpy.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = numpy.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix.transpose() - h
        weights += alpha * data_matrix.transpose() * error
    return weights


def stoc_grad_ascent0(data_list, label_list, num_iter=150):
    """
    得到随机梯度上升算法中的权值w
    :param data_list:列表，每个元素也是列表，表示一个数据的所有特征值
    :param labellist:列表，每个元素是数据的类别
    :param num_iter:整数，迭代次数
    :return:array，梯度上升算法的权值
    """
    m, n = numpy.shape(data_list)
    alpha = 0.01
    weights = numpy.ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_list[rand_index] * weights))
            error = label_list[rand_index] - h
            weights += alpha * error * data_list[rand_index]
            del (data_index[rand_index])
    return weights


def plot_best_fit(weights):
    """
    画出决策边界
    :param weights:array,梯度上升算法的权值
    """
    data_list, label_list = load_data_set()
    data_array = numpy.array(data_list)
    n = numpy.shape(data_array)[0]
    x_code1 = []
    y_code1 = []
    x_code2 = []
    y_code2 = []
    for i in range(n):
        if int(label_list[i]) == 1:
            x_code1.append(data_array[i][1])
            y_code1.append(data_array[i][2])
        else:
            x_code2.append(data_array[i][1])
            y_code2.append(data_array[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_code1, y_code1, s=30, c='red', marker='s')
    ax.scatter(x_code2, y_code2, s=30, c='green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify_vector(x, weights):
    """
    对某个特征向量进行分类
    :param x:array，待分类特征向量
    :param weights:array，权值
    :return:0或1，分类结果
    """
    prob = sigmoid(sum(x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    """
    对数据进行分类，计算错误率
    :return:浮点数，错误率
    """
    file_train = open('horseColicTraining.txt')
    file_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in file_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stoc_grad_ascent0(numpy.array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    for line in file_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(numpy.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = float(error_count / num_test_vec)
    print 'the error rate is: %f' % error_rate
    return error_rate


def main():
    numpy.seterr(all='warn')
    colic_test()


if __name__ == '__main__':
    main()