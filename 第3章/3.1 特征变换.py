# 代码 3-1

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
# 离差标准化
def MaxMinScale(data):
    m_scale = (data-data.min())/(data.max()-data.min())
    return m_scale
data_m_scale = MaxMinScale(data)
print('离差标准化之前的前5行数据为：\n', data.iloc[0:5, :], '\n',
      '离差标准化之后的前5行数据为：\n', data_m_scale.iloc[0:5, :])

# 标准差标准化
def StandarScale(data):
    s_scale = (data-data.mean())/(data.std())
    return s_scale
data_s_scale = StandarScale(data)
print('标准差标准化之前的前5行数据为：\n', data.iloc[0:5, :], '\n',
      '标准差标准化之后的前5行数据为：\n', data_s_scale.iloc[0:5, :])

# 小数定标标准化
def DecimalScale(data):
    k = np.ceil(np.log10(data.abs().max()))
    d_scale = data/(10**k)
    return d_scale
data_d_scale = DecimalScale(data)
print('小数定标标准化之前的前5行数据为：\n', data.iloc[0:5, :], '\n',
      '小数定标标准化之后的前5行数据为：\n', data_d_scale.iloc[0:5, :]),



# 代码 3-2

# 对数函数转换
def LogNorm(data):
    l_norm = np.log10(data)
    return l_norm
data_l_norm = LogNorm(data)
print('对数函数转换前的前5行数据为：\n', data.iloc[0:5, :], '\n',
      '对数函数转换后的前5行数据为：\n', data_l_norm.iloc[0:5, :])

# 反正切函数转换
import math
def TanNorm(data):
    t_norm = pd.DataFrame(np.zeros([len(data), len(data.columns)]))
    for i in range(len(data)):
        for j in range(len(data.columns)):
            t_norm.iloc[i, j] = math.atan(data.iloc[i, j])*2/np.pi
    return t_norm
data_t_norm = TanNorm(data)
print('反正切函数转换前的前5行数据为：\n', data.iloc[0:5, :], '\n',
      '反正切函数转换后的前5行数据为：\n', data_t_norm.iloc[0:5, :])



# 代码 3-3

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 2, 0], [0, 1, 1], [1, 0, 2]])
print('[0,0,0]独热编码结果为：\n', enc.transform([[0, 0, 0]]).toarray(), '\n',
      '[0,1,2]独热编码结果为：\n', enc.transform([[0, 1, 2]]).toarray(), '\n',
      '[1,2,3]独热编码结果为：\n', enc.transform([[1, 2, 3]]).toarray())



# 代码 3-4

sepal = pd.cut(data.iloc[:, 1], 3)
print('花萼宽度离散化后的3条记录分布为：\n', sepal.value_counts())



# 代码 3-5

def SameRateCut(data, k):
    w = data.quantile(np.arange(0, 1+1.0/k, 1.0/k))
    data = pd.cut(data, w)
    return data
result = SameRateCut(data.iloc[:, 1], 3).value_counts()
print('花萼宽度等频法离散化后各个类别数目分布状况为：', '\n', result)



# 代码 3-6

def KmeanCut(data, k):
    from sklearn.cluster import KMeans  # 引入KMeans
    kmodel = KMeans(n_clusters=k, n_jobs=3)  # 建立模型，n_jobs是并行数
    kmodel.fit(data.reshape((len(data), 1)))  # 训练模型
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)  # 输出聚类中心并排序
    w = c.rolling(2).mean()  # 相邻两项求中点，作为边界点
    w = pd.DataFrame([0]+list(w[0])+[data.max()])  # 把首末边界点加上
    w.fillna(value=c.min(), inplace=True)
    w = list(w.iloc[:, 0])
    # w=[0]+list(w[0])+[data.max()]  # 把首末边界点加上
    data = pd.cut(data, w)
    return data
result = KmeanCut(np.array(data.iloc[:, 1]), 3).value_counts()
print('花萼宽度聚类离散化后各个类别数目分布状况为：', '\n', result)	

