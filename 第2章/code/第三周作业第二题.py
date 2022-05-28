import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

on_h = pd.read_excel("D:/校用/大三下/人工智能/深度学习-第3周作业/深度学习-第3周作业/data/周期性分析.xlsx", index_col = None)
holiday_data = on_h.loc[on_h['工作/公休日']=='公休日', ['日期', '销售量']]
workday_data = on_h.loc[on_h['工作/公休日']!='公休日', ['日期', '销售量']]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
fig = plt.figure(figsize=(12, 6))  # 设置画布
plt.xlim((0, 80))  # 设置x轴
plt.plot(on_h.index, on_h.iloc[:, 1], color = 'yellow', linestyle='-.', 
         label='黑色-工作日')
plt.plot(holiday_data.index[:3], holiday_data.iloc[:3, 1], color = 'blue', 
         label='红色-公休日')
plt.plot(holiday_data.index[4:], holiday_data.iloc[4:, 1], color = 'blue')
for a, b in zip(holiday_data.index, holiday_data.iloc[:, 1]):
    plt.text(a, b, '公休日')  # 添加注释

plt.text(41, 6874, '工作日')
plt.text(42, 8954, '工作日')
plt.text(43, 11021, '工作日')
plt.ylabel('销售量')
plt.legend(['黄色-工作日', '蓝色-公休日'])
plt.title('节假日影响')
plt.show()