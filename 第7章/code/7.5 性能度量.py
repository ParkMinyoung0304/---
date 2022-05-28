# 代码 7-5

from sklearn.metrics import mean_absolute_error  # 平均绝对误差 
from sklearn.metrics import median_absolute_error  # 中值绝对误差
from sklearn.metrics import explained_variance_score  # 可解释方差
from sklearn.metrics import r2_score  # R方值
import pandas as pd

data = pd.read_excel('../tmp/new_reg_data_GM11_revenue.xls')  # 读取数据
data = data.set_index('Unnamed: 0')
mean_ab_error = mean_absolute_error(data.loc[range(1994, 2014), 'y'], 
                                    data.loc[range(1994,2014), 'y_pred'], 
                                    multioutput = 'raw_values')
median_ab_error = median_absolute_error(data.loc[range(1994, 2014), 'y'], 
                                        data.loc[range(1994, 2014), 'y_pred'])
explain_var_score = explained_variance_score(data.loc[range(1994, 2014), 'y'], 
                                            data.loc[range(1994, 2014), 'y_pred'], 
                                            multioutput = 'raw_values')
r2 = r2_score(data.loc[range(1994, 2014), 'y'], 
              data.loc[range(1994, 2014), 'y_pred'], 
              multioutput = 'raw_values')
print('平均绝对误差：', mean_ab_error, '\n', 
      '中值绝对误差：', median_ab_error, '\n', 
      '可解释方差：', explain_var_score, '\n', 
      'R方值:', r2)


