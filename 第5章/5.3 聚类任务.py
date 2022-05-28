# 代码 5-3

from sklearn import datasets
from sklearn.cluster import KMeans
# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 构建并训练K均值模型
kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
print('K均值模型为：\n', kmeans)

import matplotlib.pyplot as plt
# 获取模型聚类结果
y_pre = kmeans.predict(x)
# 绘制iris原本的类别
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# 绘制kmeans聚类结果
plt.scatter(x[:, 0], x[:, 1], c=y_pre)
plt.show()

from sklearn.metrics import jaccard_similarity_score, fowlkes_mallows_score, adjusted_rand_score, davies_bouldin_score
print('K均值聚类模型的Jaccard系数：', jaccard_similarity_score(y, y_pre))
print('K均值聚类模型的FM系数：', fowlkes_mallows_score(y, y_pre))
print('K均值聚类模型的调整Rand指数：', adjusted_rand_score(y, y_pre))
print('K均值聚类模型的DB指数：', davies_bouldin_score(x, kmeans.labels_))



# 代码 5-4

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 绘制iris原本的类别
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title('iris数据集')
plt.show()

# 使用LVQ类构建LVQ聚类模型并获取原型向量
import LVQ
lvq = LVQ.LVQ()
lvq.fit(x, y)
vector = lvq.vector_array

# 绘制获取的原型向量
fig = plt.figure(1)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.scatter(vector[:, 0], vector[:, 1], marker='^', c='r')
plt.title('LVQ聚类原型向量')
plt.show()




# 代码 5-5

from sklearn import datasets
from sklearn.mixture import GaussianMixture
# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 绘制样本数据
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('iris数据集', size=17)
plt.show()

# 构建聚类数为3的GMM模型
gmm = GaussianMixture(n_components=3).fit(x)
print('GMM模型：\n', gmm)

# 获取GMM模型聚类结果
gmm_pre = gmm.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=gmm_pre)
plt.title('GMM聚类', size=17)
plt.show()

from sklearn.metrics import jaccard_similarity_score, fowlkes_mallows_score, adjusted_rand_score, davies_bouldin_score
print('GMM聚类模型的Jaccard系数：', jaccard_similarity_score(y, gmm_pre))
print('GMM聚类模型的FM系数：', fowlkes_mallows_score(y, gmm_pre))
print('GMM聚类模型的调整Rand指数：', adjusted_rand_score(y, gmm_pre))
print('GMM聚类模型的DB指数：', davies_bouldin_score(x, gmm_pre))



# 代码 5-6

from sklearn.cluster import DBSCAN
# 生成两簇非凸数据
x1, y2 = datasets.make_blobs(n_samples=1000, n_features=2,
                             centers=[[1.2, 1.2]], cluster_std=[[.1]],
                             random_state=9)
# 一簇对比数据
x2, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
x = np.concatenate((x1, x2))
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

# 生成DBSCAN模型
dbs = DBSCAN(eps=0.1, min_samples=12).fit(x)
print('DBSCAN模型:\n', dbs)

# 绘制DBSCAN模型聚类结果
ds_pre = dbs.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ds_pre)
plt.title('DBSCAN', size=17)
plt.show()



# 代码 5-7

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 单链接层次聚类
clusing_ward = AgglomerativeClustering(n_clusters=3).fit(x)
print('单链接层次聚类模型为：\n', clusing_ward)

# 绘制单链接聚类结果
cw_ypre = AgglomerativeClustering(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=cw_ypre)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title('单链接聚类', size=17)
plt.show()

from sklearn.metrics import jaccard_similarity_score, fowlkes_mallows_score, adjusted_rand_score, davies_bouldin_score
print('单链接层次聚类模型的Jaccard系数：', jaccard_similarity_score(y, cw_ypre))
print('单链接层次聚类模型的FM系数：', fowlkes_mallows_score(y, cw_ypre))
print('单链接层次聚类模型的调整Rand指数：', adjusted_rand_score(y, cw_ypre))
print('单链接层次聚类模型的DB指数：', davies_bouldin_score(x, cw_ypre))

