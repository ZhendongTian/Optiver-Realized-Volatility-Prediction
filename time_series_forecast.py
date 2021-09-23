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

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

num_classes = len(np.unique(y_train))

idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]

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

X = pd.DataFrame(tail_trade_example.groupby('time_id')['price'].apply(pd.Series.tolist).tolist())
y = mask_df_joined['tag']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

num_classes = len(np.unique(y_train))

idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]
y_train[y_train == False] = 0
y_test[y_test == False] = 0

y_train[y_train == True] = 1
y_test[y_test == True] = 1

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape= X_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)








############Training LSTM using last n records
import pandas as pd
import pickle
import numpy as np
##1. Loading data
trade_example_all =  pd.read_parquet('trade_train.parquet')
with open(r"df_joined.pickle", "rb") as input_file:
    df_joined = pickle.load(input_file)

##2. Feature Engineering
trade_example_all['avg_order_size'] = trade_example_all['size']/trade_example_all['order_count']
#Remove extreme values of avg_order_size top 1%
mask = trade_example_all['avg_order_size'] < np.percentile(trade_example_all['avg_order_size'],99)
trade_example_all = trade_example_all[mask]
trade_example_all.reset_index(inplace=True)
##3. Preprocess
#3.1 Normalize column avg_order_size MinMax normlize
max_value = trade_example_all['avg_order_size'].max()
min_value = trade_example_all['avg_order_size'].min()
trade_example_all['avg_order_size'] = (trade_example_all['avg_order_size'] - min_value) / (max_value - min_value)

#Only keep last n records for every stock - time_id pair.
##Count the number of rows group by (stock_id,time_id)
size = trade_example_all.groupby(['stock_id','time_id']).size()

##Taking n = 20
last_n = 20
mask = size >= last_n
mask = mask[mask==True]
true_id = mask.index.tolist()
##mask_trade_example now contains time_id where trades are >= than 20.
trade_example_all_mask = pd.Series(list(zip(trade_example_all['stock_id'], trade_example_all['time_id']))).isin(true_id).values
trade_example_all_larger = trade_example_all[trade_example_all_mask]
##trade_example_all_larger now only contains trade record that have length >= 20.

############Training LSTM using n most prominent records.