# 代码 4-8

# 加载所需的函数
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# 加载breast_cancer数据
cancer = load_breast_cancer()
cancer_data = cancer.data
cancer_target = cancer.target
# 划分训练集和测试集
cancer_data_train, cancer_data_test, cancer_target_train, cancer_target_test = \
    train_test_split(cancer_data, cancer_target, test_size=0.2, random_state=123)
# 建立神经网络模型
# 双隐层网络结构，第一层隐层有20个神经元，第二层隐层有25个神经元
model_network = MLPClassifier(hidden_layer_sizes=(20, 27), random_state=123)
model_network.fit(cancer_data_train, cancer_target_train)
print("建立的神经网络模型为：\n", model_network)

# 预测测试集结果
cancer_target_test_pred = model_network.predict(cancer_data_test)
print('预测前20个结果为：\n', cancer_target_test_pred[:20])
print('预测前20个结果为：\n', cancer_target_test[:20])
# 评价分类模型的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("神经网络模型预测的准确率为：\
", accuracy_score(cancer_target_test, cancer_target_test_pred))
print("神经网络模型预测的精确率为：\
", precision_score(cancer_target_test, cancer_target_test_pred))
print("神经网络模型预测的召回率为：\
", recall_score(cancer_target_test, cancer_target_test_pred))
print("神经网络模型预测的F1值为：\
", f1_score(cancer_target_test, cancer_target_test_pred))

# 绘制ROC曲线
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# 求出ROC曲线的x轴和y轴
fpr, tpr, thresholds = roc_curve(cancer_target_test, cancer_target_test_pred)
# 求出auc值
print("神经网络预测结果的auc值为", auc(fpr, tpr))
plt.figure(figsize=(10, 6))
plt.title("ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(fpr, tpr)
# plt.show() 如果没画出图像请执行这行代码
