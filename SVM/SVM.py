#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import random
import numpy
import math


class OptStruct:
    '''
    储存SVM算法要用到的变量的结构体
    '''

    def __init__(self, data_mat, label_mat, C, toler, k_tup):
        self.data_mat = data_mat
        self.label_mat = label_mat
        self.C = C
        self.toler = toler
        self.m, self.n = numpy.shape(data_mat)
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = numpy.mat(numpy.zeros((self.m, 2)))
        self.w = numpy.zeros((self.n, 1))
        self.k = numpy.mat(numpy.zeros((self.m, self.m)))
        self.k_tup = k_tup
        for i in range(self.m):
            self.k[:, i] = kernel_tran(self.data_mat, self.data_mat[i, :], k_tup)


def load_data_set(filename):
    '''
    读取文件中数据
    :param filename:文件名称
    :return data_list:列表，元素为二维数组，表示数据的两个特征值
    :return label_list:列表，元素为浮点型，表示对应的数据的类标签(1或-1)
    '''
    data_list = []
    label_list = []
    fr = open(filename)
    fr.readline()
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        label_list.append(float(line_array[0]))
        for i in range(1, len(line_array)):
            data_list.append(float(line_array[i]))
    return data_list, label_list


def select_rand_alpha(i, m):
    '''
    已知第一个alpha，在其他alpha中随机选一个构成一对
    :param i:当前选择第i个alpha
    :param m:共有m个alpha
    :return:随机选择的不为i的alpha
    '''
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    '''
    使alpha的值在H和L之间
    :param aj:alpha的值
    :param H:上界
    :param L:下界
    :return:调整后alpha的值
    '''
    if aj > H:
        return H
    elif aj < L:
        return L
    else:
        return aj


def calc_Ek(struct, k):
    '''
    计算SMO中Ek的值
    :param struct:OptStruct类，储存SMO算法用到的变量
    :param k:整形，要求的第k个E
    :return:浮点型，Ek的值
    '''
    fXk = float(numpy.multiply(struct.alphas, struct.label_mat).T * struct.k[:, k] \
                + struct.b)
    Ek = fXk - float(struct.label_mat[k])
    return Ek


def select_j(i, Ei, struct):
    '''
    选择第二个alpha
    :param i:第一个alpha的下标
    :param Ei:Ei
    :param struct:OptStruct类，储存SMO算法用到的变量
    :return　j:第二个alpha的下标
    :return Ej:Ej
    '''
    j = -1
    max_diff_E = 0
    Ej = 0
    struct.e_cache[i] = [1, Ei]
    valid_ecache_list = numpy.nonzero(struct.e_cache[:, 0].A)[0]
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:
            if k != i:
                Ek = calc_Ek(struct, k)
                diff_E = abs(Ei - Ek)
                if diff_E > max_diff_E:
                    max_diff_E = diff_E
                    j = k
                    Ej = Ek
    else:
        j = select_rand_alpha(i, struct.m)
        Ej = calc_Ek(struct, j)
    return j, Ej


def update_Ek(struct, k):
    '''
    更新struct中ecache[k]的值
    :param struct:OptStruct类，储存SMO算法用到的变量
    :param k:要更新的E的下标
    '''
    Ek = calc_Ek(struct, k)
    struct.e_cache[k] = [1, Ek]


def sign(i):
    if i > 0:
        return 1
    else:
        return -1


def judge(struct):
    '''
    测试分类器性能
    :param struct:OptStruct类，储存SMO算法用到的变量
    '''
    data_mat = numpy.mat(struct.data_mat)
    sv_index = numpy.nonzero(struct.alphas.A > 0)[0]
    sv_mat = data_mat[sv_index]
    sv_label = numpy.mat(struct.label_mat)[sv_index]
    print 'there are %d support vectors' % numpy.shape(sv_index)[0]
    m, n = numpy.shape(struct.data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_tran(sv_mat, data_mat[i, :], struct.k_tup)
        predict = kernel_eval.T * numpy.multiply(sv_label, numpy.mat(struct.alphas)[sv_index]) + struct.b
        if sign(predict) != sign(struct.label_mat[i]):
            error_count += 1.0
            print 'ERROR'
            print i
    print 'error rate = %f' % (error_count / m)


def kernel_tran(x, y, k_tup):
    '''
    核函数
    :param x:核函数的输入x
    :param y:输入y
    :param k_tup:表示核函数的信息
    :return:核函数的值
    '''
    m, n = numpy.shape(x)
    k = numpy.mat(numpy.zeros((m, 1)))
    if k_tup[0] == 'lin':
        k = x * y.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            temp = x[j, :] - y
            k[j] = temp * temp.T
        k = numpy.exp(k / (-2 * k_tup[1] ** 2))
    else:
        print k_tup[0]
        print 'that kernel is not recognized'
    return k


def smo_simple(data_list, label_list, C, toler, max_iter):
    '''
    简化的SMO算法，每次优化的alpha随机选择，eta为0时不做优化
    :param data_list:列表，每个元素为数组，表示一个数据的特征向量
    :param label_list:列表，每个元素为一个数据的类标签
    :param C:SMO算法中的惩罚参数
    :param toler:精度
    :param max_iter:最大迭代次数,如果连续max_iter次迭代alpha都没有改变则退出
    :return b:学习后的SVM参数b
    :return alphas:列表，参数alpha的集合
    '''
    data_mat = numpy.mat(data_list)
    label_mat = numpy.mat(label_list).transpose()
    b = 0
    m, n = numpy.shape(data_mat)
    alphas = numpy.mat(numpy.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pair_changed = 0
        for i in range(m):
            fXi = float(numpy.multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            Ei = fXi - float(label_mat[i])
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                j = select_rand_alpha(i, m)
                fXj = float(numpy.multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print 'L=H'
                    continue
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * data_mat[i, :].T \
                      - data_mat[j, :] * data_mat[j, :].T
                if eta >= 0:
                    print 'eta=0'
                    continue
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old < 0.00001):
                    print 'j not moving enough'
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pair_changed += 1
                print 'iter:%d,i:%d,pairs changed:%d' % (iter, i, alpha_pair_changed)
        if alpha_pair_changed == 0:
            iter += 1
        else:
            iter = 0
        print 'iteration number:%d' % iter
    return b, alphas


def inner_loop(i, struct):
    '''
    已知第一个alpha，用SMO算法训练SVM
    :param i:整形，第一个alpha的下标
    :param struct:OptStruct类，储存SMO算法用到的变量
    :return :1或0，1代表更新了alpha，0代表没有更新alpha
    '''
    Ei = calc_Ek(struct, i)
    if ((struct.label_mat[i] * Ei < -struct.toler) and (struct.alphas[i] < struct.C)) or (
                (struct.label_mat[i] * Ei > struct.toler) and (struct.alphas[i] > 0)):
        j, Ej = select_j(i, Ei, struct)
        alpha_i_old = struct.alphas[i].copy()
        alpha_j_old = struct.alphas[j].copy()
        if struct.label_mat[i] != struct.label_mat[j]:
            L = max(0, struct.alphas[j] - struct.alphas[i])
            H = min(struct.C, struct.C + struct.alphas[j] - struct.alphas[i])
        else:
            L = max(0, struct.alphas[j] + struct.alphas[i] - struct.C)
            H = min(struct.C, struct.alphas[j] + struct.alphas[i])
        if L == H:
            print 'L=H'
            return 0
        eta = 2.0 * struct.k[i, j] - struct.k[i, i] - struct.k[j, j]
        if eta >= 0:
            print 'eta=0'
            return 0
        struct.alphas[j] -= struct.label_mat[j] * (Ei - Ej) / eta
        struct.alphas[j] = clip_alpha(struct.alphas[j], H, L)
        update_Ek(struct, j)
        if abs(struct.alphas[j] - alpha_j_old < 0.00001):
            print 'j not moving enough'
            return 0
        struct.alphas[i] += struct.label_mat[j] * struct.label_mat[i] * (alpha_j_old - struct.alphas[j])
        update_Ek(struct, i)
        b1 = -Ei - struct.label_mat[i] * struct.k[i, i] * (struct.alphas[i] - alpha_i_old) - \
             struct.label_mat[j] * struct.k[j, i] * (struct.alphas[j] - alpha_j_old) + struct.b
        b2 = -Ej - struct.label_mat[i] * struct.k[i, j] * (struct.alphas[i] - alpha_i_old) - \
             struct.label_mat[j] * struct.k[j, j] * (struct.alphas[j] - alpha_j_old) + struct.b
        if (0 < struct.alphas[i]) and (struct.C > struct.alphas[i]):
            struct.b = b1
        elif (0 < struct.alphas[j]) and (struct.C > struct.alphas[j]):
            struct.b = b2
        else:
            struct.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo(data_list, label_list, C, toler, max_iter, k_tup):
    '''
    简化的SMO算法，每次优化的alpha随机选择，eta为0时不做优化
    :param data_list:列表，每个元素为数组，表示一个数据的特征向量
    :param label_list:列表，每个元素为一个数据的类标签
    :param C:SMO算法中的惩罚参数
    :param toler:精度
    :param max_iter:最大迭代次数,如果连续max_iter次迭代alpha都没有改变则退出
    :param k_tup:表示核函数的信息
    :return struct:学习后的MO算法用到的变量
    '''
    struct = OptStruct(numpy.mat(data_list), numpy.mat(label_list).transpose(), C, toler, k_tup)
    iter = 0
    entire_set = True
    alpha_pair_changed = 0
    while iter < max_iter and (alpha_pair_changed > 0 or entire_set):
        alpha_pair_changed = 0
        if entire_set:
            for i in range(struct.m):
                alpha_pair_changed += inner_loop(i, struct)
            iter += 1
        else:
            non_bound = numpy.nonzero((struct.alphas.A > 0) * (struct.alphas.A < C))[0]
            for i in non_bound:
                alpha_pair_changed += inner_loop(i, struct)
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pair_changed == 0:
            entire_set = True
    return struct


def main():
    data_list, label_list = load_data_set('train.csv')
    k1 = 1
    struct = smo(data_list, label_list, 200, 0.001, 10000, ('rbf', k1))
    judge(struct)


if __name__ == '__main__':
    main()
