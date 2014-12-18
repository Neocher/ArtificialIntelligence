# -*-coding:utf-8-*-
from numpy import *
from os import listdir
import operator
import matplotlib.pyplot as plt


# 绘制图像
def draw(dating_data_matrix, dating_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_matrix[:, 1], dating_data_matrix[:, 2], 15.0 * array(dating_labels),
               15.0 * array(dating_labels))
    plt.show()


# 从文件读取数据
def file_to_matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_matrix = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        if cmp(list_from_line[-1], 'didntLike') == 0:
            class_label_vector.append(1)
        elif cmp(list_from_line[3], 'smallDoses') == 0:
            class_label_vector.append(2)
        elif cmp(list_from_line[3], 'largeDoses') == 0:
            class_label_vector.append(3)
        index += 1
    return return_matrix, class_label_vector


# 归一化数据
def auto_norm(data_set):
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = max_values - min_values
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_values, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_values


# KNN算法
def classify(index, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_matrix = tile(index, (data_set_size, 1)) - data_set
    sq_diff_matrix = diff_matrix ** 2
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 测试分类效果
def dating_class_test():
    ratio = 0.1
    dating_data_matrix, dating_labels = file_to_matrix('datingTestSet.txt')
    norm_matrix, ranges, min_Values = auto_norm(dating_data_matrix)
    m = norm_matrix.shape[0]
    num_of_test = int(m * ratio)
    error_count = 0
    for i in range(num_of_test):
        classifier_result = classify(norm_matrix[i, :], norm_matrix[num_of_test:m, :], dating_labels[num_of_test:m], 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1
    print 'the total error rate is: %f' % (float(error_count) / float(num_of_test))


# 将数字图像转化为向量
def img_to_vector(filename):
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_string = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_string[j])
    return return_vector


#检测数字图像
def handwriting_class_test():
    hw_labels = []
    training_file_name = 'trainingDigits'
    training_file_list = listdir(training_file_name)
    m = len(training_file_list)
    training_matrix = zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        class_name_str = int(filename_str.split('_')[0])
        hw_labels.append(class_name_str)
        training_matrix[i, :] = img_to_vector(training_file_name + '/' + filename_str)
    test_file_name = 'testDigits'
    test_file_list = listdir(test_file_name)
    m_test = len(test_file_list)
    error_count = 0
    for i in range(m_test):
        filename_str = test_file_list[i]
        class_name_str = int(filename_str.split('_')[0])
        vector_under_test = img_to_vector(test_file_name + '/' + filename_str)
        classifier_result = classify(vector_under_test, training_matrix, hw_labels, 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_name_str)
        if class_name_str != classifier_result:
            error_count += 1
    print 'the total error rate is: %f' % (float(error_count) / float(m_test))


handwriting_class_test()