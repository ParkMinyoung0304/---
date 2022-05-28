import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)  #python 不以科学计数法输出的方法

data = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第2周作业/深度学习-第2周作业/异常值检测.xlsx')
data = np.array(data)

# 利用箱型图的四分位距（IQR）对异常值进行检测

Percentile = np.percentile(data, [0, 25, 50, 75, 100])  # 得出百分位数

IQR = Percentile[3] - Percentile[1]  # 箱型图  四分位距

MAX = Percentile[3]+IQR*1.5  #  上限值

MIN = Percentile[1]-IQR*1.5  # 下限值

# 打印输出样本数据

sample_data=[i for i in data ]

print(len(sample_data),'个样本数据如下：\n', sample_data,'\n')

# 判断异常值————大于上限值或小于下限值

abnormal = [i for i in data if i >MAX or i < MIN] 

print('IQR检测出的异常值有:', len(abnormal),"个\n")
print('分别为：\n', abnormal,"\n")
