# 代码2-1

import pandas as pd
data = pd.read_excel('../data/data.xlsx')
print('data中元素是否为空值的布尔型DataFrame为：\n', data.isnull())
print('data中元素是否为非空值的布尔型DataFrame为：\n', data.notnull())

print('data中每个特征对应的非空值数为：\n', data.count())
print('data中每个特征对应的缺失率为：\n', 1-data.count()/len(data))



# 代码2-2

import numpy as np
array = (51, 2618.2, 2608.4, 2651.9, 3442.1, 3393.1, 3136.1, 3744.1, 
         6607.4, 4060.3, 3614.7, 3295.5, 2332.1, 2699.3, 3036.8, 
         865, 3014.3, 2742.8, 2173.5)      
# 利用箱型图的四分位距（IQR）对异常值进行检测
Percentile = np.percentile(array, [0, 25, 50, 75, 100])  # 计算百分位数
IQR = Percentile[3] - Percentile[1]  # 计算箱型图四分位距
UpLimit = Percentile[3]+IQR*1.5  # 计算临界值上界
arrayownLimit = Percentile[1]-IQR*1.5  # 计算临界值下界
# 判断异常值，大于上界或小于下界的值即为异常值
abnormal = [i for i in array if i >UpLimit or i < arrayownLimit] 
print('箱型图的四分位距（IQR）检测出的array中异常值为：\n', abnormal)
print('箱型图的四分位距（IQR）检测出的异常值比例为：\n', len(abnormal)/len(array))

# 利用3sigma原则对异常值进行检测
array_mean = np.array(array).mean()  # 计算平均值
array_sarray = np.array(array).std()  # 计算标准差
array_cha = array - array_mean  # 计算元素与平均值之差
# 返回异常值所在位置
ind = [i for i in range(len(array_cha)) if np.abs(array_cha[i])>array_sarray]
abnormal = [array[i] for i in ind]  # 返回异常值
print('3sigma原则检测出的array中异常值为：\n', abnormal)
print('3sigma原则检测出的异常值比例为：\n', len(abnormal)/len(array))








