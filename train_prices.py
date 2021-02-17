import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


df = pd.read_csv('data.csv')

df.info()
df.isnull().sum()

df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['sqft_living'] = df['sqft_living'].fillna(df['sqft_living'].median())
df['floors'] = df['floors'].fillna(df['floors'].median())
df['sqft_above'] = df['sqft_above'].fillna(df['sqft_above'].median())
df['sqft_basement'] = df['sqft_basement'].fillna(df['sqft_basement'].median())

#df.iloc[:,0:6] = df.iloc[:,0:6].apply(pd.to_numeric)
#df.iloc[:,12] = df.iloc[:,12].apply(pd.to_numeric)

x = df.iloc[:,0:6].values
y = df.iloc[:,12].values

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam())

model.fit(x, y, epochs=10)

t_model = 'price_model.h5'
model.save(t_model)
                      
