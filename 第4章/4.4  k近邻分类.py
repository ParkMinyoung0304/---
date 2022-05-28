# 代码 4-3

# 加载需要的函数
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()  # 加载数据
data = iris.data  # 属性列
target = iris.target  # 标签列
# 划分训练集、测试集
traindata, testdata, traintarget, testtarget = \
    train_test_split(data, target, test_size=0.2, random_state=123)
model_knn = KNeighborsClassifier(n_neighbors=5)  # 确定算法参数
model_knn.fit(traindata, traintarget)  # 拟合数据
print("建立的knn模型为：\n", model_knn)

# 预测测试集结果
testtarget_pre = model_knn.predict(testdata)
print('前20条记录的预测值为：\n', testtarget_pre[:20])
print('前20条记录的实际值为：\n', testtarget_pre[:20])

# 求出预测准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("预测结果准确率为：", accuracy_score(testtarget, testtarget_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(testtarget, testtarget_pre))

