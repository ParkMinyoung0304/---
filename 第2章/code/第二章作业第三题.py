import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# 绘制饼图
greens = pd.read_excel('D:/校用/大三下/人工智能/深度学习-第2周作业/深度学习-第2周作业修改版/画图.xlsx', index_col = None)
plt.pie(greens.loc[:, '盈利'], labels = greens.loc[:, '商品类别'], autopct='%1.2f%%') #注意修改为loc
plt.rcParams['font.sans-serif'] = ['SimHei']     # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号 
plt.title('8类商品盈利分布（饼图）')
plt.show()


# 绘制柱形图
plt.bar(greens.loc[:, '商品类别'], greens.loc[:, '盈利'], color='red')    #color='red' 设置颜色为红色
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('8类商品盈利分布（柱形图）')
plt.xlabel('商品类别')
plt.ylabel('盈利/元')
plt.show()
