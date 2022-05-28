# 代码 4-9

# 加载需要的函数
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()  # 加载数据
data = wine.data  # 属性列
target = wine.target  # 标签列
# 划分训练集、测试集
traindata, testdata, traintarget, testtarget = \
    train_test_split(data, target, test_size=0.2, random_state=1234)
model_rf = RandomForestClassifier()  # 确定随机森林参数
model_rf.fit(traindata, traintarget)  # 拟合数据
print("建立的随机森林模型为：\n", model_rf)

# 预测测试集结果
testtarget_pre = model_rf.predict(testdata)
print('前20条记录的预测值为：\n', testtarget_pre[:20])
print('前20条记录的实际值为：\n', testtarget[:20])

# 求出预测结果的准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("预测结果准确率为：", accuracy_score(testtarget, testtarget_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(testtarget, testtarget_pre))




# 代码 4-10

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()  # 加载数据
data = wine.data  # 属性列
target = wine.target  # 标签列
# 划分训练集、测试集
traindata, testdata, traintarget, testtarget = \
    train_test_split(data, target, test_size=0.2, random_state=1234)
model_gbm = GradientBoostingClassifier()
model_gbm.fit(traindata, traintarget)  # 训练模型
print("建立的梯度提升决策树模型为：\n", model_gbm)

# 预测测试集结果
testtarget_pre = model_gbm.predict(testdata)
print('前20条记录的预测值为：\n', testtarget_pre[:20])
print('前20条记录的实际值为：\n', testtarget[:20])

# 求出预测结果的准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("预测结果准确率为：", accuracy_score(testtarget, testtarget_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(testtarget, testtarget_pre))



# 代码 4-11

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier  # scikit-learn为0.22版本或更高版本

X, y = load_iris(return_X_y=True)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
# 模型训练
clf.fit(X_train, y_train)
print("建立的Stacking分类模型为：\n", clf)

# 预测测试集结果
clf_pre = clf.predict(X_test)
print('前20条记录的预测值为：\n', clf_pre[:20])
print('前20条记录的实际值为：\n', y_test[:20])

# 求出预测结果的准确率和混淆矩阵
from sklearn.metrics import accuracy_score, confusion_matrix
print("预测结果准确率为：", accuracy_score(y_test, clf_pre))
print("预测结果混淆矩阵为：\n", confusion_matrix(y_test, clf_pre))
