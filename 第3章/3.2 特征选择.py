# 代码 3-7

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
x, y = iris.data, iris.target
print('特征选择前数据集形状为：', x.shape)
x_new = SelectKBest(chi2, k=2).fit_transform(x, y)
print('特征选择后数据集形状为：', x_new.shape)



# 代码 3-8

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

names = iris.feature_names
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=1)  # 选择剔除1个
rfe.fit(x, y)
print('剔除排名为：\n', sorted(zip(map(lambda x: round(x, 5), rfe.ranking_), names)))



# 代码 3-9

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

print('L1正则化前数据集形状为：', x.shape)
lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
x_new = model.transform(x)
print('L1正则化后数据集形状为：', x_new.shape)



# 代码 3-10

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

# 导入数据
face = misc.face(gray=True)
# 对数据进行映射和采样
face = face / 255.0
face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
face = face / 4.0
height, width = face.shape
print('数据行数：',height)
print('数据特征数：',width)

# 对照片的右半部分加上噪声
distorted = face.copy()
distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
# 对图片的左半部分提取patch
patch_size = (7, 7)  # patch大小
data = extract_patches_2d(distorted[:, :width // 2], patch_size)
data = data.reshape(data.shape[0], -1)
# zscore标准化
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
# 创建MiniBatchDictionaryLearning模型
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500).fit(data)
print('字典学习：\n',dico)

# 调用components_属性返回获取的字典
V = dico.components_
# 画出V中的字典
plt.figure(figsize=(4.2, 4))     # 定义图片大小，4.2英寸宽，4英寸高
# 循环画出100个字典
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.subtitle('Dictionary learned from face patches',fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23) #left, right, bottom, top, wspace, hspace.
plt.show()



# 代码 3-11

from sklearn.feature_extraction.image import reconstruct_from_patches_2d
# 定义函数用于对比字典学习前后的效果
def show_with_diff(image, reference, title):
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference
    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

# 画出图像及噪声
show_with_diff(distorted, face, 'Distorted image')
plt.show()

# 提取图像含有噪声的右半部进行字典学习。
data = extract_patches_2d(distorted[:, width // 2:], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
# 四种不同的字典表示策略
transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

# 循环绘出4种字典表示策略的结果
reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = face.copy()
    # 通过set_params方法对参数进行设置
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    # 通过transform方法对数据进行字典表示
    code = dico.transform(data)
    # code矩阵乘V得到复原后的矩阵patches
    patches = np.dot(code, V)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
    # 通过reconstruct_from_patches_2d函数将patches重新拼接回图片
    reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(
        patches, (height, width // 2))
    show_with_diff(reconstructions[title], face,title )
    plt.show()
