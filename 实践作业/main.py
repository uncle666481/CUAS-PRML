# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 加载MNIST数据集
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(int)

# 数据预处理：标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 线性SVM模型
svm = make_pipeline(SVC(kernel='linear', C=1))
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Bagging模型：使用SVM作为基础学习器
bagging_svm = BaggingClassifier(base_estimator=SVC(kernel='linear', C=1), n_estimators=10, random_state=42)
bagging_svm.fit(X_train, y_train)
bagging_svm_pred = bagging_svm.predict(X_test)
bagging_svm_accuracy = accuracy_score(y_test, bagging_svm_pred)

# 神经网络模型
mlp = MLPClassifier(hidden_layer_sizes=(512, 128), activation='relu', max_iter=20, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

# 打印结果
print(f"Linear SVM Accuracy: {svm_accuracy * 100:.2f}%")
print(f"Bagging with SVM Accuracy: {bagging_svm_accuracy * 100:.2f}%")
print(f"Neural Network Accuracy: {mlp_accuracy * 100:.2f}%")

# 绘制结果
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# SVM
axes[0].imshow(svm_pred.reshape(28,28)[0], cmap='gray')
axes[0].set_title(f"SVM Prediction: {svm_pred[0]}")
axes[0].axis('off')

# Bagging SVM
axes[1].imshow(bagging_svm_pred.reshape(28,28)[0], cmap='gray')
axes[1].set_title(f"Bagging SVM Prediction: {bagging_svm_pred[0]}")
axes[1].axis('off')

# Neural Network
axes[2].imshow(mlp_pred.reshape(28,28)[0], cmap='gray')
axes[2].set_title(f"NN Prediction: {mlp_pred[0]}")
axes[2].axis('off')

plt.show()
