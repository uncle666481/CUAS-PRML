
#MNIST 分类：SVM，Bagging 和神经网络

本项目展示了如何使用三种不同的机器学习模型对 MNIST 数据集进行分类：
- **线性支持向量机（SVM）**
- **SVM 的 Bagging（Bootstrap 聚合）**
- **神经网络（MLP - 多层感知机）**

项目的目标是比较这些模型在 MNIST 手写数字数据集上的性能，并展示如何在 Python 中使用 `scikit-learn` 实现这些算法。

## 介绍

**MNIST 数据集**是一个常用的机器学习基准数据集，专门用于手写数字分类。本项目应用了三种模型：
- **线性SVM**：一种传统的机器学习方法，通过寻找最佳超平面来分类。
- **SVM 的 Bagging**：一种集成方法，通过训练多个模型并聚合它们的预测结果来提升性能。
- **神经网络（MLP）**：一种深度学习模型，能捕捉复杂的模式，尤其适合分类任务。

这些模型将使用 MNIST 数据集进行训练和测试，评估并比较每种模型的分类性能（准确率）。

## 要求

本项目需要 Python 3 和以下库：
- `numpy`（用于数值计算）
- `scikit-learn`（用于机器学习算法和工具）
- `matplotlib`（用于结果可视化）
- `fetch_openml`（用于加载 MNIST 数据集）

你可以使用以下命令安装所需的库：

```bash
pip install numpy scikit-learn matplotlib
```

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/mnist_classification.git
   ```

2. 进入项目目录：
   ```bash
   cd mnist_classification
   ```

3. 安装依赖（如果依赖未安装）：
   ```bash
   pip install -r requirements.txt
   ```

## 使用

要运行代码，直接执行 `mnist_classification.py` 脚本：

```bash
python mnist_classification.py
```

### 代码说明：

- **数据预处理**：加载 MNIST 数据并进行标准化处理，将数据拆分为训练集和测试集。
- **模型训练**： 
  - **线性 SVM**：使用线性 SVM 分类器进行训练。
  - **SVM 的 Bagging**：使用 Bagging 方法集成多个 SVM 分类器。
  - **神经网络**：使用多层感知机（MLP）进行训练。
- **评估**：输出每个模型在测试集上的准确率，并可视化模型的预测结果。

## 结果

脚本会输出每个模型的分类准确率：
- **线性 SVM**
- **SVM 的 Bagging**
- **神经网络**

同时，还会显示每个模型的一些预测示例图像。

示例输出：

```
线性 SVM 准确率: 98.43%
SVM 的 Bagging 准确率: 98.52%
神经网络 准确率: 97.96%
```

不同模型的准确率可能会有轻微波动，具体取决于随机种子的设置。

