import pandas as pd
import numpy as np

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

train_df_tt = pd.read_csv('C:/D_Disk/data_competition/gamer_value/data/tap_fun_train.csv', 
                       index_col=0, header=0)
print(mem_usage(train_df_tt))


train_df_tt.info()
train_df_tt.columns
dtype_dict = dict(train_df_tt.dtypes)
print(dtype_dict)
for col_name, col_type in dtype_dict.items():
    if dtype_dict[col_name]==np.int64:
        dtype_dict[col_name]=np.int8
    elif dtype_dict[col_name]==np.float64:
        dtype_dict[col_name]=np.float16
print('after change: ')
print(dtype_dict)


train_df_tt_new = pd.read_csv('C:/D_Disk/data_competition/gamer_value/data/tap_fun_train.csv', 
                               index_col=0, header=0, dtype=dtype_dict)

train_df_tt_new.info()
print(mem_usage(train_df_tt_new))