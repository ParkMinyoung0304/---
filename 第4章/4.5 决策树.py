# 代码 4-4

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 导入load_breast_cancer数据
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']
# 将数据划分为训练集测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# 训练决策树模型，
dt_model = DecisionTreeClassifier(criterion='entropy')
dt_model.fit(x_train, y_train)
print('建立的决策树模型为：\n', dt_model)

# 预测测试集结果
test_pre = dt_model.predict(x_test)
print('前10条记录的预测值为：\n', test_pre[:10])
print('前10条记录的实际值为：\n', y_test[:10])

# 求出预测准确率和混淆矩阵
from sklearn.metrics import accuracy_score,confusion_matrix
print("预测结果准确率为：", accuracy_score(y_test, test_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(y_test, test_pre))



# 代码 4-5

# 加载需要的函数
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()  # 加载数据
data = iris.data  # 属性列
target = iris.target  # 标签列
# 划分训练集、测试集
traindata, testdata, traintarget, testtarget = \
    train_test_split(data, target, test_size=0.2, random_state=123)
model_dtc = DecisionTreeClassifier()  # 确定决策树参数
model_dtc.fit(traindata, traintarget)  # 拟合数据
print("建立的决策树模型为：\n", model_dtc)

# 预测测试集结果
testtarget_pre = model_dtc.predict(testdata)
print('前20条记录的预测值为：\n', testtarget_pre[:20])
print('前20条记录的实际值为：\n', testtarget[:20])

# 求出预测准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("预测结果准确率为：", accuracy_score(testtarget, testtarget_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(testtarget, testtarget_pre))
