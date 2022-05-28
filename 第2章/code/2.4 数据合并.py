# 代码2-12

import pandas as pd
Sheet_stack1 = pd.read_excel('../data/Sheet_stack1.xlsx', index_col=0)
Sheet_stack2 = pd.read_excel('../data/Sheet_stack2.xlsx', index_col=0)
# 横向堆叠
Sheet_in = pd.concat([Sheet_stack1, Sheet_stack2], axis=1, join='inner')
Sheet_out = pd.concat([Sheet_stack1, Sheet_stack2], axis=1, join='outer')
print('外连接横向堆叠后的数据框为：\n', Sheet_out)
print('内连接横向堆叠后的数据框：\n', Sheet_in)
print('外连接横向堆叠后的数据框大小为：', Sheet_out.shape)
print('内连接横向堆叠后的数据框大小为：', Sheet_in.shape)

# 纵向堆叠
# 利用concat函数
Sheet_in_0 = pd.concat([Sheet_stack1, Sheet_stack2], axis=0, join='inner')
Sheet_out_0 = pd.concat([Sheet_stack1, Sheet_stack2], axis=0, join='outer')
print('外连接纵向堆叠后的数据框为：\n', Sheet_out_0)
print('内连接纵向堆叠后的数据框：\n', Sheet_in_0)
print('外连接纵向堆叠后的数据框大小为：', Sheet_out_0.shape)
print('内连接纵向堆叠后的数据框大小为：', Sheet_in_0.shape)

# 利用append方法
Sheet_append = Sheet_stack1.append(Sheet_stack2)
print('append方法合并数据后数据框为：\n', Sheet_append)
print('append方法合并数据后数据框大小为：', Sheet_append.shape)



# 代码2-

# 主键合并
import pandas as pd
Sheet_key1 = pd.read_excel('../data/Sheet_key1.xlsx', index_col=0)
Sheet_key2 = pd.read_excel('../data/Sheet_key2.xlsx', index_col=0)
print('主键合并前Sheet_key1的大小为：', Sheet_key1.shape, '\n', 
      '主键合并前Sheet_key2的大小为：', Sheet_key2.shape)

Sheet_key = pd.merge(Sheet_key1, Sheet_key2, left_on='key', right_on = 'key')
print('主键合并后数据框为：\n', Sheet_key, '\n', 
      '主键合并后数据框大小为：', Sheet_key.shape)
