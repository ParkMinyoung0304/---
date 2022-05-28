# 代码 9-6

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 客户分群雷达图
cluster_center = pd.read_csv('../tmp/cluster_centers.csv')
labels = pd.read_csv('../tmp/labels.csv')
cluster_center.columns = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC']  # 将聚类中心放在数据框中
cluster_center.index = labels.drop_duplicates().iloc[:, 0]  # 将样本类别作为数据框索引
labels = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC']
lstype = ['-','--',':','-.']
legen = ['客户群' + str(i + 1) for i in cluster_center.index]  # 客户群命名，作为雷达图的图例
kinds = list(cluster_center.iloc[:, 0])

# 由于雷达图要保证数据闭合，因此再添加ZL列 ，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])
# 分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint = False)
angle = np.concatenate((angle, [angle[0]]))

# 绘图
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111, polar = True)  # 以极坐标的形式绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 

# 画线
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2, label=kinds[i])
# 添加属性标签
ax.set_thetagrids(angle * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.savefig('D:/Pictures/机器学习实战/9-6.1.jpg', dpi=2080) #指定分辨率保存
plt.show()

