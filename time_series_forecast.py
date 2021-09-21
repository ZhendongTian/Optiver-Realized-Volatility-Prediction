# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:41:05 2021

@author: 78182
"""


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")



####using trade_example and df_joined



##Count the number of rows group by time_id
size = trade_example.groupby('time_id').size()

##Get average number of samples mean = 32
mask = size >= 20
mask = mask[mask==True]
true_id = mask.index.tolist()
##mask_trade_example now contains time_id where trades are >= than 20
mask_trade_example = trade_example[trade_example['time_id'].isin(true_id)]

###Now only get the last 20 timestamps
g = mask_trade_example.groupby('time_id')
tail_trade_example = g.tail(20)
tail_time_id = tail_trade_example['time_id'].unique()
tail_row_id = [f'0-%d'%t_id for t_id in tail_time_id]

###prepare df_joined for prediction
df_joined['tag'] = df_joined['pred'] > df_joined['target']
mask_df_joined = df_joined[df_joined['row_id'].isin(tail_row_id)]

X = tail_trade_example.groupby('time_id')['price'].apply(pd.Series.tolist).tolist()
y = 