import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

data_path = 'C:/D_Disk/data_competition/caini_xihuan/data/RSdata/'
df_test = pd.read_csv(data_path + 'test.csv')
df_train = pd.read_csv(data_path + 'train.csv')

print('df_train.shape is ', df_train.shape,
      'df_train head(5) is ', df_train.head(5))
print('df_test.shape is ', df_test.shape)


small_data_file = 'C:/github_base/data_competition/caini_xihuan/small_data.csv'
df_small = pd.read_csv(small_data_file)
print('df_small is ', df_small.shape, df_small.head(5))

user_u = list(sorted(df_train.uid.unique()))
item_u = list(sorted(df_train.iid.unique()))

row = df_train.uid.astype('category', categories=user_u).cat.codes
col = df_train.iid.astype('category', categories=item_u).cat.codes
data = df_train['score'].tolist()

sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))
df_pivoted = pd.SparseDataFrame([ pd.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0) 
                              for i in np.arange(sparse_matrix.shape[0]) ], 
                       index=user_u, columns=item_u, default_fill_value=0)

print('df_pivoted head is ', df_pivoted.head(50))

