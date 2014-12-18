#!/usr/bin/python2
# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import tree

decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def get_num_leafs(my_tree):
    """
    获取一棵树的叶子节点个数，树用字典表示
    :param my_tree:要分析的树
    :return:叶子节点个数
    """
    num_leafs = 0
    second_dict = my_tree[my_tree.keys()[0]]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取一棵树的最大深度，树用字典表示
    :param my_tree:要分析的树
    :return:树的深度
    """
    max_depth = 0
    second_dict = my_tree[my_tree.keys()[0]]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            temp_depth = 1 + get_tree_depth(second_dict[key])
        else:
            temp_depth = 1
        if temp_depth > max_depth:
            max_depth = temp_depth
    return max_depth


def plot_node(node_txt, center_pt, patent_pt, node_type):
    """
    绘制箭头及节点
    :param node_txt:节点上的内容
    :param center_pt:子节点坐标
    :param patent_pt:父节点坐标
    :param node_type:节点类型
    """
    global ax1
    ax1.annotate(node_txt, xy=patent_pt, xycoords='axes fraction', xytext=center_pt, textcoords='axes fraction',
                 va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """
    在父子节点间添加信息
    :param cntr_pt:子节点坐标
    :param parent_pt:父节点坐标
    :param txt_string:要添加的信息
    """
    global ax1
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2 + cntr_pt[1]
    ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    """
    画出my_tree表示的树
    :param my_tree:要画出的树
    :param parent_pt:树的根节点坐标
    :param node_txt:根节点的备注
    """
    global ax1
    global total_d
    global total_w
    global x_off
    global y_off
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = my_tree.keys()[0]
    cntr_pt = (x_off + (1.0 + float(num_leafs)) / 2.0 / total_w, y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    y_off -= 1.0 / total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            x_off += 1.0 / total_w
            plot_node(second_dict[key], (x_off, y_off), cntr_pt, leaf_node)
            plot_mid_text((x_off, y_off), cntr_pt, str(key))
    y_off += 1.0 / total_d


def create_plot(my_tree):
    """
    根据my_tree画出树
    :param my_tree:要画的树
    """
    global ax1
    global total_d
    global total_w
    global x_off
    global y_off
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    ax1 = plt.subplot(111, frameon=False, **axprops)
    total_w = float(get_num_leafs(my_tree))
    total_d = float(get_tree_depth(my_tree))
    x_off = -0.5 / total_w
    y_off = 1.0
    plot_tree(my_tree, (0.5, 1.0), '')
    plt.show()


def main():
    data_set, labels = tree.create_data_set()
    my_tree = tree.create_tree(data_set, labels)
    create_plot(my_tree)


if __name__ == '__main__':
    main()