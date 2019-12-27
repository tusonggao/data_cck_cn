import os
import gc
import time

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

int_types = ["uint8", "int8", "int16", 'int32', 'int64']
int_types_min_max_list = []
for it in int_types:
    int_types_min_max_list.append((it, np.iinfo(it).min, np.iinfo(it).max))
print('int_types_min_max_list: ', int_types_min_max_list)

float_types = ['float16', 'float32', 'float64']
#float_types = ['float32', 'float64']
float_types_min_max_list = []
for ft in float_types:
    float_types_min_max_list.append((ft, np.finfo(ft).min, np.finfo(ft).max))

print('float_types_min_max_list: ', float_types_min_max_list)

def generate_numeric_dtypes(df, store_file_path=None):
    dtypes_dict = dict(df.dtypes)
    dtypes_dict_new = dict()
    
    for col_name, d_t in dtypes_dict.items():
        if is_numeric_dtype(df[col_name])==False:
            continue
        if d_t==np.float64 or d_t==np.float32:
            for ft, min_v, max_v in float_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    dtypes_dict_new[col_name] = ft
                    break
        elif d_t==np.int64 or d_t==np.int32 or d_t==np.int16:
            for it, min_v, max_v in int_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    dtypes_dict_new[col_name] = it
                    break
        else:
            dtypes_dict_new[col_name] = d_t
    
    dtypes_df = pd.DataFrame.from_dict(
                     {'col_name': list(dtypes_dict_new.keys()),
                      'best_dtype': list(dtypes_dict_new.values())}
                )
    dtypes_df = dtypes_df.set_index('col_name')
    print('dtypes_df.shape: ', dtypes_df.shape)
    if store_file_path is not None:
        dtypes_df.to_csv(store_file_path)
    return dtypes_dict_new

def convert_numeric_dtypes(df):
    print('original df info: ')
    print(df.info(memory_usage='deep'))
    dtypes_dict = dict(df.dtypes)
    
    start_t = time.time()
    for col_name, d_t in dtypes_dict.items():
        if is_numeric_dtype(df[col_name])==False:
            continue
        if d_t==np.float64 or d_t==np.float32:
            for ft, min_v, max_v in float_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    print(col_name, 'prev_type: ', d_t, 'best_type: ', ft)
                    df[col_name] = df[col_name].astype(ft)
                    break
        elif d_t==np.int64 or d_t==np.int32 or d_t==np.int16:
            for it, min_v, max_v in int_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    print(col_name, 'prev_type: ', d_t, 'best_type: ', it)
                    df[col_name] = df[col_name].astype(it)
                    break
    
    gc.collect()
    print('convert cost time: ', time.time()-start_t)
    
    print('after convert df info: ')
    print(df.info(memory_usage='deep'))
    return df


start_t = time.time()
data_dir_path = 'C:/D_Disk/data_competition/home_credit/data/'
merged_df = pd.read_csv(data_dir_path + '/processed/merged_df.csv', index_col=0)
print('merged_df.shape is ', merged_df.shape,
      'cost time: ', time.time()-start_t)
print(merged_df.info(memory_usage='deep'))

#generate_numeric_dtypes(
#    merged_df,
#    'C:/D_Disk/data_competition/home_credit/data/processed/best_dtypes.csv'
#)

convert_numeric_dtypes(merged_df)

del merged_df
gc.collect()

#merged_X_new_new.dtypes.to_csv('./best_dtypes.csv')

#best_dtypes_df = pd.read_csv(
#    'C:/D_Disk/data_competition/home_credit/data/processed/best_dtypes.csv',
#    index_col=0
#)
#dtypes_dict = dict(best_dtypes_df['best_dtype'])

