# 实验1：ID3 决策树

## 简介
这是一个简单的ID3决策树的Python实现，用于从数据中构建决策树，进行分类预测。ID3（Iterative Dichotomiser 3）是一种经典的决策树学习算法，用于处理分类问题。

## 项目结构
- `ID3.py`: 包含了构建决策树的核心函数代码。
- `train.csv`: 包含训练数据的CSV文件。训练数据用于构建决策树。
- `predict.csv`: 包含预测数据的CSV文件。您可以使用训练后的决策树来对这些数据进行分类预测。
- `ID3.ipynb`：包含了读取数据，训练决策树，预测结果的步骤，运行该程序即可。

## 如何使用

### 1. 安装依赖项
确保您已经安装了以下Python库：

```bash
pip install numpy pandas pydot
```

### 2. 数据准备
- 创建您自己的训练数据文件 `train.csv`，并根据您的问题定义适当的特征和目标列。
- 创建预测数据文件 `predict.csv`，包含您希望对其进行分类预测的样本。

### 3. 运行代码
运行 `ID3.ipynb` ，它将构建决策树并执行分类预测。训练后的决策树将根据训练数据自动构建。

## 项目函数

 `calculate_entropy(data)`

- 描述：计算给定数据集的熵（Entropy）。
- 参数：`data` - 数据集。
- 返回值：熵的值。

$$
H(S) = -\sum_{i=1}^{n} p_i \log_2(p_i) 

$$

`calculate_information_gain(data, attribute)`

- 描述：计算根据给定属性分割数据后的信息增益（Information Gain）。
- 参数：`data` - 数据集，`attribute` - 属性名称。
- 返回值：信息增益的值。

$$

IG(S, A) = H(S) - \sum_{a \in A} \frac{\lvert S_a \rvert}{\lvert S \rvert} \cdot H(S_a) 

$$

 `choose_best_attribute(data, attributes)`

- 描述：从给定属性列表中选择最佳属性，以最大化信息增益。
- 参数：`data` - 数据集，`attributes` - 属性列表。
- 返回值：最佳属性的名称。

 `build_decision_tree(data, attributes)`

- 描述：根据给定数据和属性列表构建决策树。
- 参数：`data` - 数据集，`attributes` - 属性列表。
- 返回值：构建的决策树。

 `predict(tree, sample)`

- 描述：使用构建好的决策树对样本进行分类预测。
- 参数：`tree` - 构建好的决策树，`sample` - 样本数据。
- 返回值：预测结果。

## 注意事项

- 请确保您的数据集与代码中的列名和数据格式匹配。
- 决策树的构建和预测是基于ID3算法实现的，因此它可能不适用于所有问题，特别是在处理连续型数据时需要额外的处理。
