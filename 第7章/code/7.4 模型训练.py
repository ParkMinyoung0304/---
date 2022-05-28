# 代码 7-3

# 自定义灰色预测函数 
def GM11(x0):  # x0为矩阵形式
    import numpy as np
    x1 = x0.cumsum()  # 1-AGO序列
    # 紧邻均值（MEAN）生成序列
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis = 1)
    Yn = x0[1:].reshape((len(x0)-1, 1))
    # 计算参数
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn) 
    # 还原值
    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (
            x0[0] - b / a) * np.exp(-a * (k - 2)) 
    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)
    # 返回灰色预测函数、a、b、首项、方差比、小残差概率
    return f, a, b, x0[0], C, P

import pandas as pd
import numpy as np

new_reg_data = pd.read_csv('../tmp/new_reg_data.csv')  # 读取经过特征选择后的数据
data = pd.read_csv('../data/data.csv')  # 读取总的数据
new_reg_data.index = range(1994, 2014)
new_reg_data.loc[2014] = None
new_reg_data.loc[2015] = None
Accuracy = []  # 存放灰色预测模型精度
l = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']
for i in l:
    f = GM11(new_reg_data.loc[range(1994, 2014), i].as_matrix())[0]
    new_reg_data.loc[2014, i] = f(len(new_reg_data) - 1)  # 2014年预测结果
    new_reg_data.loc[2015, i] = f(len(new_reg_data))  # 2015年预测结果
    new_reg_data[i] = new_reg_data[i].round(2)  # 保留两位小数
    C = GM11(new_reg_data.loc[range(1994, 2014), 'x1'].as_matrix())[4]
    P = GM11(new_reg_data.loc[range(1994, 2014), 'x1'].as_matrix())[5]
    if P>0.95 and C<0.35:
        Accuracy.append('好')
    elif 0.8<P<=0.95 and 0.35<=C<0.5:
        Accuracy.append('合格')
    elif 0.7<P<=0.8 and 0.5<=C<0.65:
        Accuracy.append('勉强合格')
    else :
        Accuracy.append('不合格')

new_reg_data = new_reg_data.iloc[:, 1:]
new_reg_data.loc['模型精度', :] = Accuracy
outputfile = '../tmp/new_reg_data_GM11.xls'  # 灰色预测后保存的路径
# 提取财政收入列，合并至新数据框中
y = list(data['y'].values)
y.extend([np.nan, np.nan])
new_reg_data.loc[range(1994, 2016),'y'] = y
new_reg_data.to_excel(outputfile)  # 结果输出
# 预测结果展示
print('预测结果为：\n', new_reg_data.loc[[2014, 2015, '模型精度'], :])



# 代码 7-4

from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt

data = pd.read_excel('../tmp/new_reg_data_GM11.xls')  # 读取数据
data = data.set_index('Unnamed: 0')
data = data.drop(index = '模型精度')
feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']  # 特征所在列
data_train = data.loc[range(1994, 2014)].copy()  # 取2014年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean) / data_std  # 数据标准化
x_train = data_train[feature].as_matrix()  # 特征数据
y_train = data_train['y'].as_matrix()  # 标签数据
linearsvr = LinearSVR(random_state=123)  # 调用LinearSVR()函数
linearsvr.fit(x_train, y_train)

# 预测2014年和2015年财政收入，并还原结果。
x = ((data[feature] - data_mean[feature]) / data_std[feature]).as_matrix()
data[u'y_pred'] = linearsvr.predict(x) * data_std['y'] + data_mean['y']
outputfile = '../tmp/new_reg_data_GM11_revenue.xls'
data.to_excel(outputfile)
print('真实值与预测值分别为：\n', data[['y', 'y_pred']])

print('预测图为：', data[['y', 'y_pred']].plot(style = ['b-o', 'r-*']))  # 画出预测结果图
plt.xlabel('年份')
plt.xticks(range(1994,2015,2))
