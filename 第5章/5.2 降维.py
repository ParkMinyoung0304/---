# 代码 5-1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 生成样本数据集
x, y = make_moons(n_samples=100, random_state=233)
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title("样本数据")
plt.show()

# 使用PCA对样本数据进行降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
# 绘制降维结果并与原样本数据对比
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(x_pca[y == 0, 0], x_pca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_pca[y == 0, 0], np.zeros((50, 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(x_pca[y == 1, 0], np.zeros((50, 1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_title("样本数据")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('X_pca')
ax[1].set_title("降维后样本数据")
plt.tight_layout()
plt.show()



# 代码 5-2

from sklearn.decomposition import KernelPCA

# 使用KernelPCA对样本数据进行降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
x_kpca = kpca.fit_transform(x)
# 绘制降维结果并与原样本数据对比
kfig, kx = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
kx[0].scatter(x[y == 0, 0], x[y == 0, 1], color='red', marker='^', alpha=0.5)
kx[0].scatter(x[y == 1, 0], x[y == 1, 1], color='blue', marker='o', alpha=0.5)
kx[1].scatter(x_kpca[y == 0, 0], x_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
kx[1].scatter(x_kpca[y == 1, 0], x_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
kx[0].set_xlabel('X')
kx[0].set_ylabel('Y')
kx[0].set_title("样本数据")
kx[1].set_ylim([-1, 1])
kx[1].set_yticks([])
kx[1].set_xlabel('X_kpca')
kx[1].set_title("降维后样本数据")
plt.tight_layout()
plt.show()
