__author__ = 'lufo'
# -*-coding:utf-8-*-
from math import log
import operator
import pickle


def create_data_set():
    """
    建立测试用的数据集
    :return:数据集和它标签的集合
    """
    file_reader = open('lenses.txt')
    data_set = [inst.strip().split('\t') for inst in file_reader.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算香农熵
    :param data_set: 要计算的数据集
    :return:该数据集的香农熵
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vector in data_set:
        current_label = feat_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0
    for key in label_counts:
        prob = float(label_counts[key]) / float(num_entries)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    按照特征划分数据集
    :param data_set:待划分的数据集
    :param axis:划分依据的特征
    :param value:返回的数据集的特征的值
    :return:返回划分后特征值为value的数据集
    """
    return_data_set = []
    for feat_vector in data_set:
        if feat_vector[axis] == value:
            reduced_feat_vector = feat_vector[:axis]
            reduced_feat_vector.extend(feat_vector[axis + 1:])
            return_data_set.append(reduced_feat_vector)
    return return_data_set


def choose_best_feature_to_split(data_set):
    """
    选择用于划分数据集的最优特征
    :param data_set:要被划分的数据集
    :return:最优特征的下标
    """
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        feature_set = set(feature_list)
        new_entropy = 0.0
        for value in feature_set:
            sub_data_set = split_data_set(data_set, i, value)
            prob = float(len(sub_data_set)) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    返回一个列表中出现次数最多的元素
    :param class_list:要统计的列表
    :return:出现次数最多的元素
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_list.iteritem(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count


def create_tree(data_set, labels):
    """
    使用ID3算法构造数据集的决策树
    :param data_set:用来构造决策树的数据集
    :param labels:数据的属性的集合
    :return:数据集的决策树，为字典类型
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set) == 1:
        return majority_cnt(class_list)
    best_feature_index = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature_index]
    my_tree = {best_feature_label: {}}
    del (labels[best_feature_index])
    feature_values_list = [example[best_feature_index] for example in data_set]
    feature_values_set = set(feature_values_list)
    for value in feature_values_set:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature_index, value),
                                                         sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    根据决策树分类数据
    :param input_tree:字典，分类依据的决策树
    :param feat_labels:列表，决策树分类标签的集合
    :param test_vec:列表，用于分类的数据，每个元素表示feat_labels对应的特征的值
    :return:字符串，分类结果
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    """
    将决策树储存为文件
    :param input_tree:字典，要储存的树
    :param filename:字符串，储存为文件的名字
    """
    file_writer = open(filename, 'w')
    pickle.dump(input_tree, file_writer)
    file_writer.close()


def grab_tree(filename):
    """
    从文件中读取决策树
    :param filename:文件名
　　 """
    file_reader = open(filename)
    return pickle.load(file_reader)


def main():
    my_data, labels = create_data_set()
    print my_data
    my_tree = create_tree(my_data, labels)
    print my_tree


if __name__ == '__main__':
    main()