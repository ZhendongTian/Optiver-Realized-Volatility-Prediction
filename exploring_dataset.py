# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:33:36 2021

@author: 78182
"""


import pandas as pd
book_train = pd.read_parquet('book_train.parquet')
top = book_train[:1000]

import pandas as pd
import numpy as np
import plotly.express as px
train = pd.read_csv('train.csv')
train.head()

book_example = pd.read_parquet('book_train.parquet/stock_id=0')
trade_example =  pd.read_parquet('trade_train.parquet/stock_id=0')

stock_id = '0'
book_example = book_example[book_example['time_id']==5]
book_example.loc[:,'stock_id'] = stock_id
trade_example = trade_example[trade_example['time_id']==5]
trade_example.loc[:,'stock_id'] = stock_id

book_example['wap'] = (book_example['bid_price1'] * book_example['ask_size1'] +
                                book_example['ask_price1'] * book_example['bid_size1']) / (
                                       book_example['bid_size1']+ book_example['ask_size1'])
    
fig = px.line(book_example, x="seconds_in_bucket", y="wap", title='WAP of stock_id_0, time_id_5')
fig.show()

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()

book_example.loc[:,'log_return'] = log_return(book_example['wap'])
book_example = book_example[~book_example['log_return'].isnull()]

fig = px.line(book_example, x="seconds_in_bucket", y="log_return", title='Log return of stock_id_0, time_id_5')
fig.show()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))
realized_vol = realized_volatility(book_example['log_return'])
print(f'Realized volatility for stock_id 0 on time_id 5 is {realized_vol}')


def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] =(df_book_data['bid_price1'] * df_book_data['ask_size1']+df_book_data['ask_price1'] * df_book_data['bid_size1'])  / (
                                      df_book_data['bid_size1']+ df_book_data[
                                  'ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]


def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] =(df_book_data['bid_price1'] * df_book_data['ask_size1']+df_book_data['ask_price1'] * df_book_data['bid_size1'])  / (
                                      df_book_data['bid_size1']+ df_book_data[
                                  'ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]



def past_realized_volatility_per_stock(list_file,prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        df_past_realized = pd.concat([df_past_realized,
                                     realized_volatility_per_time_id(file,prediction_column_name)])
    return df_past_realized


import glob
list_order_book_file_train = glob.glob('book_train.parquet/*')


df_past_realized_train = past_realized_volatility_per_stock(list_file=list_order_book_file_train,
                                                           prediction_column_name='pred')


train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_joined = train.merge(df_past_realized_train[['row_id','pred']], on = ['row_id'], how = 'left')


############Analyse The Difference Between target and pred

df_joined['diff'] = df_joined['pred'] - df_joined['target']
df_joined['diff_abs'] = abs(df_joined['pred'] - df_joined['target'])

trade_example['mean_order_size'] = trade_example['size']/trade_example['order_count']
trade_example['price_change'] = trade_example['price'].pct_change(1)




####Self-defined features
#1.no of big order
#2.no of abnormal size
#3.no of abnormal order_count
#4.

###
#Scatter plot pred-diff_abs
#
from scipy import stats
df_scatter = df_joined[['pred','diff_abs']]
df_scatter[(np.abs(stats.zscore(df_scatter[0])) < 3)]
import numpy as np
import matplotlib.pyplot as plt

df_scatter = df_joined[['pred','diff_abs','diff']]
df_scatter = df_scatter[df_scatter['pred'] < 0.006]
df_scatter = df_scatter[df_scatter['diff_abs'] < 0.0018]
df_scatter = df_scatter[df_scatter['diff'] < 0.0018]
df_scatter = df_scatter[df_scatter['diff'] > -0.0018]
plt.hist2d(df_scatter['pred'],df_scatter['diff_abs'],bins=100)
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df_scatter['pred'], df_scatter['diff_abs'])
plt.show()

import numpy as np
import mpl_scatter_density
import matplotlib.pyplot as plt

# Generate fake data

x = df_scatter['pred']
y = df_scatter['diff']

# Make the plot - note that for the projection option to work, the
# mpl_scatter_density module has to be imported above.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(x, y)
ax.set_xlim(0, 0.006)
ax.set_ylim(-0.0018, 0.0018)
fig.savefig('gaussian.png')