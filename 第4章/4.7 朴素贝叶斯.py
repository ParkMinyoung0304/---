# 代码 4-7

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯算法

iris = load_iris()  # 加载iris数据集
data = pd.DataFrame(iris.data, columns=iris.feature_names)  # 提取数据并存入数据框s
target = pd.DataFrame(iris.target, columns=['Species'])  # 提取标签并存入数据框q
# 划分训练集和测试集（训练集：测试集 = 8:2）
x_train, x_test, y_train, y_test = \
train_test_split(data, target, random_state=1234, test_size=0.2)
model = GaussianNB()  # 实例化
# 建立模型并进行训练
model.fit(x_train, y_train)
pre = list(model.predict(x_test))  # 利用测试集进行预测
print('预测结果：', pre)

# 计算模型的准确率
from sklearn.metrics import accuracy_score
print('准确率', accuracy_score(y_test, pre))

