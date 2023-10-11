# _*_ coding : utf-8 _*_
# @Time : 2023/10/8 19:49
# @Author : Black
# @File : ID3
# @Project : hm1

# 构建ID#决策树


import numpy as np
import os
import pydot

# 设置当前环境变量
os.environ["PATH"] += os.pathsep + 'E:/Python/Anaconda/Lib/graphviz/bin/'


def calculate_entropy(data):
    labels = data['weather']
    unique_labels = labels.unique()
    entropy = 0
    total_samples = len(data)

    for label in unique_labels:
        label_count = len(data[data['weather'] == label])
        label_probability = label_count / total_samples
        entropy -= label_probability * np.log2(label_probability)

    return entropy


def calculate_information_gain(data, attribute):
    entropy_before_split = calculate_entropy(data)
    unique_values = data[attribute].unique()
    total_samples = len(data)
    weighted_entropy_after_split = 0

    for value in unique_values:
        subset = data[data[attribute] == value]
        subset_samples = len(subset)
        weighted_entropy_after_split += (subset_samples / total_samples) * calculate_entropy(subset)

    information_gain = entropy_before_split - weighted_entropy_after_split
    return information_gain


def choose_best_attribute(data, attributes):
    best_attribute = None
    max_information_gain = -1

    for attribute in attributes:
        information_gain = calculate_information_gain(data, attribute)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute

    return best_attribute


def build_decision_tree(data, attributes):
    if len(data['weather'].unique()) == 1:
        # 如果所有样本的类别都相同，返回该类别作为叶子节点
        return data['weather'].iloc[0]
    elif len(attributes) == 0:
        # 如果没有可用属性，返回数据中最常见的类别作为叶子节点
        return data['weather'].mode().iloc[0]
    else:
        best_attribute = choose_best_attribute(data, attributes)
        tree = {best_attribute: {}}
        # print(tree)
        unique_values = data[best_attribute].unique()
        for value in unique_values:
            subset = data[data[best_attribute] == value]
            subtree = build_decision_tree(subset, [attr for attr in attributes if attr != best_attribute])
            tree[best_attribute][value] = subtree
        return tree


def predict(tree, sample):
    if isinstance(tree, str):
        # 如果当前节点是叶子节点，直接返回类别
        return tree
    else:
        # 获取当前节点的分割属性
        attribute = list(tree.keys())[0]
        value = sample[attribute]  # 获取样本中的对应属性值
        if value in tree[attribute]:
            # 如果属性值在决策树中存在，递归预测下一个节点
            next_node = tree[attribute][value]
            return predict(next_node, sample)
        else:
            # 如果属性值在决策树中不存在，返回默认类别或处理方式
            return "Unknown"  # 或者根据需求返回其他值


def visualize_tree(data, parent=None, graph=None):
    if graph is None:
        graph = pydot.Dot(graph_type='digraph')

    if isinstance(data, dict):
        for key, value in data.items():
            node = pydot.Node(key, label=str(key))
            graph.add_node(node)

            if parent is not None:
                edge = pydot.Edge(parent, node)
                graph.add_edge(edge)

            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_node = pydot.Node(sub_key, label=str(sub_key))
                    graph.add_node(sub_node)
                    edge = pydot.Edge(node, sub_node)
                    graph.add_edge(edge)
                    visualize_tree(sub_value, sub_node, graph)

    return graph
