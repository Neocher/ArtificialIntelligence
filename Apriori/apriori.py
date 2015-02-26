#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
def load_data_list():
    '''
    加载数据集
    :return:列表，数据集
    '''
    data_list = []
    fr = open('result.csv')
    for line in fr.readlines():
        data = line.strip().split(',')
        map(int, data)
        data_list.append(data)
    return data_list


def create_c1(data_list):
    '''
    创建只有一个元素的候选集列表
    :param data_list:列表 数据集
    :return:列表，元素为frozenset格式的候选集
    '''
    c1 = []
    for data in data_list:
        for item in data:
            if not [item] in c1:
                c1.append([item])
    return map(frozenset, c1)


def create_ck(lk, k):
    '''
    由Lk得到Ck
    :param lk:列表，每个元素为满足最小支持度的数据，frozenset格式
    :param k:整型，k
    :return:列表Ck
    '''
    ck = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            if l1 == l2:
                ck.append(lk[i] | lk[j])
    return ck


def scan_data_set(data_set, ck, min_support):
    '''
    得到候选集中达到最小支持度的
    :param data_set:set，数据集
    :param ck:列表，元素为frozenset格式的候选集
    :param min_support:浮点数，最小支持度
    :return can_list:列表，每个元素为满足最小支持度的数据，frozenset格式
    :return support_data:字典，保存每个数据的支持度，key为frozenset格式的数据，value为支持度
    '''
    data_count = {}
    for data in data_set:
        for can in ck:
            if can.issubset(data):
                if data_count.has_key(can):
                    data_count[can] += 1
                else:
                    data_count[can] = 1
    data_items = float(len(data_set))
    can_list = []
    support_data = {}
    for key in data_count:
        support = data_count[key] / data_items
        if support >= min_support:
            can_list.insert(0, key)
        support_data[key] = support
    return can_list, support_data


def apriori(data_list, min_support=0.5):
    '''
    apriori算法，得到所有满足最小支持度的数据
    :param data_list:列表 数据集
    :param min_support:浮点数，最小支持度
    :return l:列表，每个元素lk,lk为有k个元素的满足最小支持度的数据，frozenset格式
    :return support_data:字典，保存每个数据的支持度，key为frozenset格式的数据，value为支持度
    '''
    c1 = create_c1(data_list)
    data_set = map(set, data_list)
    l1, support_data = scan_data_set(data_set, c1, min_support)
    l = [l1]
    k = 2
    while len(l[k - 2]) > 0:
        ck = create_ck(l[k - 2], k)
        lk, support_data_k = scan_data_set(data_set, ck, min_support)
        support_data.update(support_data_k)
        l.append(lk)
        k += 1
    return l, support_data


def calc_conf(freq_set, h, support_data, big_rule_list, min_conf):
    '''
    计算自信度
    :param freq_set:一个频繁项集
    :param h:列表，长度为i的freq_set的子集的集合
    :param support_data:字典，保存每个数据的支持度，key为frozenset格式的数据，value为支持度
    :param big_rule_list:列表，保存满足自信度的数据
    :param min_conf:最小自信度
    :return:长度为i+1的满足最小自信度的freq_set的子集的集合
    '''
    pruned_h = []
    for q in h:
        conf = support_data[freq_set] / support_data[freq_set - q]
        if conf >= min_conf:
            print freq_set - q, '-->', q, 'conf:', conf
            big_rule_list.append((freq_set - q, q, conf))
            pruned_h.append(q)
    return pruned_h


def rules_from_q(freq_set, h, support_data, big_rule_list, min_conf):
    '''
    当频繁项集中数据大于2时，递归计算自信度
    :param freq_set:一个频繁项集
    :param h:列表，长度为i的freq_set的子集的集合
    :param support_data:字典，保存每个数据的支持度，key为frozenset格式的数据，value为支持度
    :param big_rule_list:列表，保存满足自信度的数据
    :param min_conf:最小自信度
    '''
    m = len(h[0])
    if len(freq_set) > (m + 1):
        h_next = create_ck(h, m + 1)
        h_next = calc_conf(freq_set, h_next, support_data, big_rule_list, min_conf)
        if len(h_next) > 1:
            rules_from_q(freq_set, h_next, support_data, big_rule_list, min_conf)


def generate_rules(l, support_data, min_conf=0.7):
    '''
    生成关联规则
    :param l:列表，频繁项集，每个元素lk,lk为有k个元素的满足最小支持度的数据，frozenset格式
    :param support_data:字典，保存每个数据的支持度，key为frozenset格式的数据，value为支持度
    :param min_conf:最小支持度
    :return:列表，包含满足关联规则的频繁项集
    '''
    big_rules_list = []
    for i in range(1, len(l)):
        for freq_set in l[i]:
            h = [frozenset([item]) for item in freq_set]
            if i == 1:
                calc_conf(freq_set, h, support_data, big_rules_list, min_conf)
            else:
                rules_from_q(freq_set, h, support_data, big_rules_list, min_conf)
    return big_rules_list


def main():
    data_list = load_data_list()
    l, support_data = apriori(data_list, 0.05)
    big_rules_list = generate_rules(l, support_data,0.1)
    print big_rules_list


if __name__ == '__main__':
    main()