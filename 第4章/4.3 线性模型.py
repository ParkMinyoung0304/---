# 代码 4-1

# 加载所需函数
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# 加载boston数据
boston = load_boston()
x = boston['data']
y = boston['target']
names = boston['feature_names']
# 将数据划分为训练集测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=125)
# 建立线性回归模型
clf = LinearRegression().fit(x_train, y_train)
print('建立的LinearRegression模型为：', '\n', clf)

# 预测测试集结果
y_pred = clf.predict(x_test)
print('预测前20个结果为：', '\n', y_pred[:20])

from sklearn.metrics import explained_variance_score, mean_absolute_error,\
mean_squared_error,median_absolute_error,r2_score
print('Boston数据线性回归模型的平均绝对误差为：', 
      mean_absolute_error(y_test, y_pred))
print('Boston数据线性回归模型的均方误差为：', 
      mean_squared_error(y_test, y_pred))
print('Boston数据线性回归模型的中值绝对误差为：',
      median_absolute_error(y_test, y_pred))
print('Boston数据线性回归模型的可解释方差值为：',
      explained_variance_score(y_test, y_pred))
print('Boston数据线性回归模型的R方值为：', 
      r2_score(y_test, y_pred))



# 代码 4-2

# 加载所需的函数
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 加载breast_cancer数据
cancer = load_breast_cancer()
cancer_data = cancer.data
cancer_target = cancer.target
# 划分训练集和测试集
cancer_data_train, cancer_data_test, cancer_target_train, cancer_target_test = \
    train_test_split(cancer_data, cancer_target, test_size=0.2, random_state=123)
# 建立逻辑回归模型
lr = LogisticRegression()
lr.fit(cancer_data_train, cancer_target_train)
print("建立的逻辑回归模型为：\n", lr)

# 预测测试集结果
cancer_target_test_pred = lr.predict(cancer_data_test)
print('预测前20个结果为：\n', cancer_target_test_pred[:20])

# 求出预测取值和真实取值一致的数目 
import numpy as np
num_accu = np.sum(cancer_target_test == cancer_target_test_pred)
print('预测对的结果数目为：', num_accu)
print('预测错的结果数目为：', cancer_target_test.shape[0]-num_accu)
print('预测结果准确率为：', num_accu/cancer_target_test.shape[0])

