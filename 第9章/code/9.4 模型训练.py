# 代码9-5

from sklearn.cluster import KMeans  # 导入KMeans算法
import matplotlib.pyplot as plt  # 导入画图库
from sklearn import metrics  # 导入计算Calinski-Harabasz指数的库
import pandas as pd
import numpy as np

airline_scale = np.load('../tmp/airline_scale.npz')['arr_0']
# 利用Calinski-Harabasz指数确定聚类数目
CH = []
for i in range(3, 6):
    model = KMeans(n_clusters = i, n_jobs=4, random_state =123).fit(airline_scale)
    labels = model.labels_
    CH.append(metrics.calinski_harabaz_score(airline_scale, labels)) 
k = CH.index(max(CH)) + 3  # 确定聚类中心数
print('最佳聚类数目', k)

# 绘制不同聚类数目与对应的Calinski-Harabasz指数折线图
x = range(3, 6)  # x为折线图中的横坐标
plt.plot(x, CH, '-xr')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
plt.title('不同聚类数目对应的Calinski-Harabasz指数')
# 构建模型
kmeans_model = KMeans(n_clusters = k, n_jobs=4, random_state=123)
fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练
# 查看聚类中心
print('聚类中心为：\n', kmeans_model.cluster_centers_)
print('保留小数点后4位后聚类中心为：\n', np.round(kmeans_model.cluster_centers_, 4))

print('样本类别标签为', kmeans_model.labels_)  # 查看样本的类别标签

# 统计不同类别样本的数目
count_class = pd.Series(kmeans_model.labels_).value_counts()
print('最终每个类别的数目为：\n', count_class)
cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_)
cluster_centers.to_csv('../tmp/cluster_centers.csv',index=False)  # 保存聚类中心
labels = pd.DataFrame(kmeans_model.labels_)
labels.to_csv('../tmp/labels.csv',index=False)  # 保存聚类类别标签
