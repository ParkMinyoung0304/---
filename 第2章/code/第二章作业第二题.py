import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第2周作业/深度学习-第2周作业修改版/直方图.xlsx')
data = np.array(data)

print("共读取"+str(len(data))+"个数据")
# 极差
Max_Min = max(data)-min(data)
print('极差为:',Max_Min, '\n')

# 分组 初始组距:15
group = round(Max_Min[0]/15)  # 确定组数
print('分组组数为', group, '\n')

# group->分点
point = np.linspace(min(data), max(data), group)
print('分点有',str(len(point)),'个,分为','\n', point,'\n')

# 分点->组距
group_gap = point[1] - point[0]
print('最终组距为', group_gap,'\n')

# 绘制频率分布表
table_frequency = pd.DataFrame(np.zeros([8, 5]), columns = ['组段', '组中值x', '频数', '频率f', '累计频率'])
print('频率分布表的初始状态为：\n', table_frequency)
f_sum = 0  # 累计频率初始值
for i in range(len(point)):
        table_frequency.loc[i, '组段'] = '['+str(np.round(point[i], 2))+','+ str(np.round(point[i]+group_gap, 2))+')'
        
        table_frequency.loc[i, '组中值x'] = np.round(np.array((point[i], point[i]+group_gap)).mean (), 2)

        table_frequency.loc[i, '频数'] = sum([pd.notnull(j) for j in data if point[i] <= j < point[i]+group_gap])

        table_frequency.loc[i, '频率f'] = table_frequency.loc[i, '频数']/len(data)

        f_sum = f_sum + table_frequency.loc[i, '频率f']

        table_frequency.loc[i, '累计频率'] = f_sum 

print('频率分布表为：\n', table_frequency)

# 计算频率与组距的比值，作为频率分布直方图的纵坐标
y = table_frequency.loc[:, '频率f']/group_gap

# 绘制频率分布直方图
fig = plt.figure(figsize=(15, 5),facecolor="yellow")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

plt.rcParams['axes.unicode_minus'] = False  # 显示负号 

ax = fig.add_subplot(111)

#设置宽度为0.8英寸
plt.bar(table_frequency.loc[:, '组段'], y, 0.8)

plt.xlabel('分布区间')

plt.ylabel('频率/组距')

plt.title('频率分布直方图')

plt.show()