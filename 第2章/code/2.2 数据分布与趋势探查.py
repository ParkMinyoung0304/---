# 代码2-3

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

sale = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第2周作业/深度学习-第2周作业修改版/深度学习-第2周作业修改版/data/直方图.xlsx')
sale = np.array(sale)
# 求极差
sale_jicha = max(sale)-min(sale)
# 分组，这里取初始组距为1000
group = round(sale_jicha[0]/16)  # 确定组数
# 根据group对数据进行切片，即决定分点
bins = np.linspace(min(sale), max(sale), group)
# 根据分点确定最终组距
zuju = bins[1] - bins[0]
print('极差为', sale_jicha, '\n', 
      '分组组数为', group, '\n', 
      '分点为：\n', bins, '\n', 
      '最终组距为', zuju)

# 绘制频率分布表
table_fre = pd.DataFrame(np.zeros([8, 5]), 
                         columns = ['组段', '组中值x', '频数', '频率f', '累计频率'])
f_sum = 0  # 累计频率初始值
for i in range(len(bins)):
        table_fre.loc[i, '组段'] = '['+str(np.round(bins[i], 2))+','+ \
        str(np.round(bins[i]+zuju, 2))+')'
        table_fre.loc[i, '组中值x'] = np.round(np.array((bins[i], 
                    bins[i]+zuju)).mean (), 2)
        table_fre.loc[i, '频数'] = sum([pd.notnull(j) for j in sale if bins[i] <= \
                    j < bins[i]+zuju])
        table_fre.loc[i, '频率f'] = table_fre.loc[i, '频数']/len(sale)
        f_sum = f_sum + table_fre.loc[i, '频率f']
        table_fre.loc[i, '累计频率'] = f_sum 
print('频率分布表为：\n', table_fre)

# 计算频率与组距的比值，作为频率分布直方图的纵坐标
y = table_fre.loc[:, '频率f']/zuju
# 绘制频率分布直方图
fig = plt.figure(figsize=(14, 4))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
ax = fig.add_subplot(111)
plt.bar(table_fre.loc[:, '组段'], y, 0.8)
plt.xlabel('分布区间')
plt.ylabel('频率/组距')
plt.title('频率分布直方图')
plt.show()



# 代码2-4

# 绘制饼图
greens = pd.read_excel('../data/greens.xlsx', index_col = None)
plt.pie(greens.loc[:, '盈利'], labels = greens.loc[:, '菜品名'], autopct='%1.2f%%')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
plt.title('10种菜品盈利分布（饼图）')
plt.show()

# 绘制柱形图
plt.bar(greens.loc[:, '菜品名'], greens.loc[:, '盈利'])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('10种菜品盈利分布（柱形图）')
plt.xlabel('菜品名')
plt.ylabel('盈利/元')
plt.show()



# 代码2-5

compare = pd.read_excel('../data/compare.xlsx', index_col=0)
ls = ['-', '--', '-.', ':']  # 线条类型
leg = compare.columns  # 图例
cl=['red', 'orange', 'green', 'blue']  # 线条颜色
plt.figure(figsize=(9, 6))  # 画布大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
# 画多条折线对比
for i in range(4):
    plt.plot(compare.iloc[:, i], linestyle=ls[i], color=cl[i], label=leg[i])
plt.title('4种菜品销售量趋势')  # 图片标题
plt.legend()  # 显示图例
plt.show



# 代码2-6

vegetable = pd.read_excel('../data/生炒菜心.xlsx', index_col=None)
ls = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5))]
leg = vegetable.columns
cl=['red', 'orange', 'green', 'blue', 'purple']
plt.figure(figsize=(9, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
for i in range(5):
    plt.plot(vegetable.iloc[:, i], linestyle=ls[i], color=cl[i], label=leg[i])
plt.title('生炒菜心不同月份之间的销售额比较')
plt.legend()
plt.show



# 代码2-7

statistic = pd.read_excel('../data/statistic.xlsx', index_col = 0)
explore = statistic.describe().T
explore['jicha'] = explore['max'] - explore['min']  # 计算极差
explore['IQR'] = explore['75%'] - explore['25%']  # 计算四分位差
explore['cv'] = explore['std']/explore['mean']
explore['std_2'] = explore['std']**2  # 计算方差
explore['median'] = np.median(statistic.iloc[:, 0])  # 计算中位数
explore['zhong'] = np.argmax(np.bincount(statistic.iloc[:, 0]))  # 计算众数
print('某菜品销售额统计量情况：\n', explore.T)



# 代码2-8

on_h = pd.read_excel('D:\校用\大三下\人工智能\《机器学习原理与实战》源代码和实验数据\第2章\data\vacation.xlsx', index_col = 0)
holiday_data = on_h.loc[on_h['holiday']=='小长假', ['date', 'on_man']]
workday_data = on_h.loc[on_h['holiday']!='小长假', ['date', 'on_man']]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
fig = plt.figure(figsize=(12, 6))  # 设置画布
plt.xlim((0, 80))  # 设置x轴
plt.plot(on_h.index, on_h.iloc[:, 1], color = 'black', linestyle='-.', 
         label='黑色-工作日')
plt.plot(holiday_data.index[:3], holiday_data.iloc[:3, 1], color = 'red', 
         label='红色-节假日')
plt.plot(holiday_data.index[4:], holiday_data.iloc[4:, 1], color = 'red')
for a, b in zip(holiday_data.index, holiday_data.iloc[:, 1]):
    plt.text(a, b, '小长假')  # 添加注释
plt.ylabel('人流量')
plt.legend(['黑色-工作日', '红色-节假日'])
plt.title('节假日影响')
plt.show()



# 代码2-9

# 帕累托分析
palt = pd.read_excel('../data/greens.xlsx ')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 绘制直方图
palt1 = palt.loc[:, ['菜品名', '盈利']]
palt1 = palt1.sort_values('盈利', ascending = False)
palt1 = palt1.set_index('菜品名')
palt1_h = palt1.plot(kind='bar')

# 绘制折线
palt2 = 1*palt1['盈利'].cumsum()/palt1['盈利'].sum()
palt2_h= palt2.plot(color = 'black', secondary_y = True, style = '-x', linewidth = 2)
palt1_h.legend(loc = 'upper center')
# 添加标注
palt2 = palt2.reset_index(drop=True)
palt3 = palt2[palt2>=0.8][0:1]
point_X = palt3.index[0]
point_Y = palt3[point_X]
# 添加注释，即85%处的标记，这里包括了指定箭头样式
plt.annotate(format(point_Y, '.2%'), xy = (point_X, point_Y),
             xytext=(point_X*0.9, point_Y*0.9), arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3, rad=.2")) 
# 设置标签
plt.ylabel('盈利（元）')
plt.ylabel('盈利（比例）')
plt.show()



# 代码2-10

# 读取菜品销售量数据
cor = pd.read_excel('../data/cor.xlsx') 
# 计算相关系数矩阵，包含了任意两个菜品间的相关系数
print('5种菜品销售量的相关系数矩阵为：\n', cor.corr())

# 绘制相关性热力图
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(8, 8))  # 设置画面大小 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
sns.heatmap(cor.corr(), annot=True, vmax=1, square=True, cmap="Blues") 
plt.title('相关性热力图')
plt.show()
