# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 22:26:24 2021

@author: 78182
"""


from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
import lightgbm as lgb

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt 
import seaborn as sns

path_root = '../'
path_data = '../'
path_submissions = '/'

target_name = 'target'
scores_folds = {}

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def get_stock_stat(stock_id : int, dataType = 'train'):
    key = ['stock_id', 'time_id', 'seconds_in_bucket']
    
    #Book features
    df_book = pd.read_parquet(os.path.join(path_data, 'book_{}.parquet/stock_id={}/'.format(dataType, stock_id)))
    df_book['stock_id'] = stock_id
    cols = key + [col for col in df_book.columns if col not in key]
    df_book = df_book[cols]
    
    df_book['wap1'] = (df_book['bid_price1'] * df_book['ask_size1'] +
                                    df_book['ask_price1'] * df_book['bid_size1']) / (df_book['bid_size1'] + df_book['ask_size1'])
    df_book['wap2'] = (df_book['bid_price2'] * df_book['ask_size2'] +
                                    df_book['ask_price2'] * df_book['bid_size2']) / (df_book['bid_size2'] + df_book['ask_size2'])
    df_book['log_return1'] = df_book.groupby(by = ['time_id'])['wap1'].apply(log_return).fillna(0)
    df_book['log_return2'] = df_book.groupby(by = ['time_id'])['wap2'].apply(log_return).fillna(0)
    
    features_to_apply_realized_volatility = ['log_return'+str(i+1) for i in range(2)]
    stock_stat = df_book.groupby(by = ['stock_id', 'time_id'])[features_to_apply_realized_volatility]\
                        .agg(realized_volatility).reset_index()

    #Trade features
    trade_stat =  pd.read_parquet(os.path.join(path_data,'trade_{}.parquet/stock_id={}'.format(dataType, stock_id)))
    trade_stat = trade_stat.sort_values(by=['time_id', 'seconds_in_bucket']).reset_index(drop=True)
    trade_stat['stock_id'] = stock_id
    cols = key + [col for col in trade_stat.columns if col not in key]
    trade_stat = trade_stat[cols]
    trade_stat['trade_log_return1'] = trade_stat.groupby(by = ['time_id'])['price'].apply(log_return).fillna(0)
    trade_stat = trade_stat.groupby(by = ['stock_id', 'time_id'])[['trade_log_return1']]\
                           .agg(realized_volatility).reset_index()
    #Joining book and trade features
    stock_stat = stock_stat.merge(trade_stat, on=['stock_id', 'time_id'], how='left').fillna(-999)
    
    return stock_stat

def get_dataSet(stock_ids : list, dataType = 'train'):

    stock_stat = Parallel(n_jobs=-1)(
        delayed(get_stock_stat)(stock_id, dataType) 
        for stock_id in stock_ids
    )
    
    stock_stat_df = pd.concat(stock_stat, ignore_index = True)

    return stock_stat_df

def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False

params_lgbm = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'objective': 'regression',
        'metric': 'None',
        'max_depth': -1,
        'n_jobs': -1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'lambda_l2': 1,
        'verbose': -1
        #'bagging_freq': 5
}


train = pd.read_csv(os.path.join(path_data, 'train.csv'))
train_stock_stat_df = get_dataSet(stock_ids = train['stock_id'].unique(), dataType = 'train')
train = pd.merge(train, train_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
print('Train shape: {}'.format(train.shape))
display(train.head(2))

test = pd.read_csv(os.path.join(path_data, 'test.csv'))
test_stock_stat_df = get_dataSet(stock_ids = test['stock_id'].unique(), dataType = 'test')
test = pd.merge(test, test_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left').fillna(0)
print('Test shape: {}'.format(test.shape))
display(test.head(2))