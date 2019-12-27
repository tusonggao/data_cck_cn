import os
import sys
import string
import time
import random
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb

current_path = os.path.split(os.path.realpath(__file__))[0]
# current_path = 'F:/github_me_repos/data_competition/kdxf2019_mobileAD/'
os.chdir(current_path)

def generate_muffle_files(file_num=100):
    start_t = time.time()

    def file_number_under_path(dirname):
        result = []  # 所有的文件
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                file_path = os.path.join(maindir, filename)  # 合并成一个完整路径
                result.append(file_path)
        return len(result)

    def gen_content(line_num=25):
        candidate_chs = string.digits + string.ascii_letters
        content = ''
        for i in range(line_num):
            content += ''.join(random.choices(candidate_chs, k=120)) + '\n'
        # print('content is ', content)
        return content

    def gen_rand_file(file_name_len=16):
        all_candidate_chs = string.digits + string.ascii_letters
        upper_letters = ''.join(set(string.ascii_uppercase) - set('AGMS'))
        print('upper_letters is ', upper_letters)
        print(all_candidate_chs)
        file_name = ''
        for i in range(file_name_len):
            if i<=2:
                file_name += random.choice(upper_letters)
            else:
                file_name += random.choice(all_candidate_chs)
        print('file_name is ', file_name)
        with open(current_path + '/temp_data/' + file_name + '.txt', 'w') as file:
            file.write(gen_content())
        return file_name

    # print('file_number is ', file_number_under_path(current_path + '/temp_data/'))

    for i in range(file_num):
        gen_rand_file()
    print('generate_muffle_files cost time ', time.time()-start_t)

generate_muffle_files()

sys.exit(0)

df_train = pd.read_csv(current_path + '/data/round1_iflyad_anticheat_traindata.txt', sep='\t')
print('df_train.shape is ', df_train.shape, df_train.head(10), list(df_train.columns))  # (1000000, 29)

df_test = pd.read_csv(current_path + '/data/round1_iflyad_anticheat_testdata_feature.txt', sep='\t')
print('df_test.shape is ', df_test.shape, df_test.head(10), list(df_test.columns))

df_merged = pd.concat([df_train, df_test])
df_merged = df_merged.sample(250000)

df_merged = df_merged.fillna(-1)

# df_merged['year'] = df_merged['nginxtime'].apply(lambda x: int(time.strftime("%Y", time.localtime(x//1000))))
# df_merged['month'] = df_merged['nginxtime'].apply(lambda x: int(time.strftime("%m", time.localtime(x//1000))))
# df_merged['day'] = df_merged['nginxtime'].apply(lambda x: int(time.strftime("%d", time.localtime(x//1000))))
df_merged['hour'] = df_merged['nginxtime'].apply(lambda x: int(time.strftime("%H", time.localtime(x//1000))))
df_merged['minute'] = df_merged['nginxtime'].apply(lambda x: int(time.strftime("%M", time.localtime(x//1000))))

del df_merged['nginxtime']

# media_cate_feature = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype']
media_cate_feature = ['ver', 'apptype']
# ip_cate_feature = ['reqrealip', 'city', 'province']
ip_cate_feature = ['city', 'province']
# device_cate_feature = ['carrier', 'os', 'osv', 'ntt', 'model', 'make', 'ppi']
device_cate_feature = []
origin_cate_list = media_cate_feature + ip_cate_feature + device_cate_feature

print('get here 111')

# 编码，加速
for i in origin_cate_list:
    df_merged[i] = df_merged[i].map(dict(zip(df_merged[i].unique(), range(0, df_merged[i].nunique()))))

print('df_merged.shape is ', df_merged.shape, df_merged.columns)

count_feature_list = []

# print('df_merged["label"] ', df_merged[['year', 'month', 'day', 'hour', 'minute', 'nginxtime']])

def feature_count(data, features=[], is_feature=True):
    print('in feature_count()')
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    nunique = []
    for i in features:
        nunique.append(data[i].nunique())
        new_feature += '_' + i.replace('add_', '')
    if len(features) > 1 and len(data[features].drop_duplicates()) <= np.max(nunique):
        print(new_feature, 'is unvalid cross feature:')
        return data
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    if is_feature:
        count_feature_list.append(new_feature)
    # print('in feature_count, data.shape: {}, features: {}, new_feature: {}'.format(
    #       data.shape, features, new_feature))
    return data



for i in origin_cate_list:
    n = df_merged[i].nunique()
    if n > 5:
        df_merged = feature_count(df_merged, [i])
        # data = feature_count(data, ['day', 'hour', i])
    else:
        print('feature: ', i, ' nunique less than 5')

print('after add feature_count, df_merged.shape is ', df_merged.shape, df_merged.columns)

cate_feature = origin_cate_list
num_feature = ['h', 'w', 'hour', 'minute'] + count_feature_list
feature = cate_feature + num_feature

# predict = data[(data.label == -1) & (data.data_type == 2)]
predict = df_merged[(df_merged.label == -1)]

predict_result = predict[['sid']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)
train_x = df_merged[df_merged.label != -1].reset_index(drop=True)
train_y = train_x.pop('label').values
base_train_csr = sparse.csr_matrix((len(train_x), 0))
base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

enc = OneHotEncoder()
for feature in cate_feature:
    print('one-hot processing feature:', feature)
    enc.fit(df_merged[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))),
                                   'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                     'csr', 'bool')
print('one-hot prepared !')

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype('float32')

predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')


print('train_csr.shape is ', train_csr.shape,
      'predict_csr.shape is ', predict_csr.shape)

print('get here 777')

# sys.exit(0)

lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=61, reg_alpha=3, reg_lambda=1,
    max_depth=-1, n_estimators=500, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.035, random_state=2018, n_jobs=10
)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)

final_f1_score = []
best_score = []

for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    print('in enumerate 888, index: ', index)
    print('len of train_index', len(train_index), 'len of test_index', len(test_index))
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=200, verbose=10)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)

    partial_pred = lgb_model.predict(train_csr[test_index], num_iteration=lgb_model.best_iteration_)
    f1 = f1_score(train_y[test_index], partial_pred, average='macro')
    final_f1_score.append(f1)
    print('f1 is ', f1)
    print('final_f1_score is ', final_f1_score)

    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    test_pred_outcome = lgb_model.predict(predict_csr, num_iteration=lgb_model.best_iteration_).astype(int)
    test_pred_outcome_from_val =  (test_pred > 0.5).astype(int)

    print('test_pred.head(20)', test_pred[:20])
    print('test_pred_outcome_from_val.head(20)', test_pred_outcome_from_val[:20])
    print('test_pred_outcome.head(20)', test_pred_outcome[:20])

    print()

    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

predict_result['predicted_score'] = predict_result['predicted_score'] / 5.0

predict_result['label'] = (predict_result['predicted_score'].values > 0.5)
predict_result['label'] = predict_result['label'].astype(int)

now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
predict_result[['sid', 'label']].to_csv(current_path + '/submission/lgb_outcome_{}.csv'.format(now_str),
                                        index=False)

print('ends here!')







