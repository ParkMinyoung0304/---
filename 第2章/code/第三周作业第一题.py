import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

compare = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第3周作业/深度学习-第3周作业/data/对比分析.xlsx', index_col=0)
ls = ['-', '--', '-.', ':',(45,(55,20))]  # 线条类型，设置虚线每段的线段长度和间隔长度，及偏移量
leg = compare.columns  # 图例
cl=['red', 'orange', 'green', 'blue','purple']  # 线条颜色
plt.figure(figsize=(20, 10))  # 画布大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
# 多条折线对比
for i in range(5):
    plt.plot(compare.iloc[:, i], linestyle=ls[i], color=cl[i], label=leg[i])
plt.title('5种菜品销售量趋势')  # 图片标题
plt.legend(loc = 'upper center')  # 显示图例居中
plt.xlabel('销售日期')  #x轴
plt.ylabel('销售份数')   #y轴
plt.show()

compare = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第3周作业/深度学习-第3周作业/data/对比分析.xlsx', index_col = 0)
explore = compare['香煎罗卜糕'].describe()
print('"香煎萝卜糕"销售量统计性分析:\n',explore)
print("极差:",explore['max'] - explore['min']) # 计算极差
print("四分位距:",explore['75%'] - explore['25%']) # 计算四分位差
print("变异差数:",explore['std']/explore['mean']) # 计算变异差数
print("方差:",explore['std']**2) # 计算方差
print("中位数:",np.median(compare.iloc[:, 0])) # 计算中位数
print("众数:",np.argmax(np.bincount(compare.iloc[:, 0]))) # 计算众数


