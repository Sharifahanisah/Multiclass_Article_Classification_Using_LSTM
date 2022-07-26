# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 01:44:18 2022

@author: HP
"""

import numpy as np 
import pandas as pd
import datetime
import pickle
import json
import os

from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential,Input
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

#%%
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')

#%% step 1) Data loading 
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

#%% step 2) data inspection

df.info
df.describe().T
df.head

df.duplicated().sum()
df.isna().sum()

print(df['text'][4])
print(df['text'][10])

#%% step 3) Data Cleaning

import re

articles = df['text']
category = df['category']

articles_backup = articles.copy()
category_backup = category.copy()  

for index, text in enumerate(articles):
    
    articles[index] = re.sub('< .*?>','',text)
    articles[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()

articles_backup = articles.copy()
category_backup = category.copy()  
    
     

#%% step 4) feature selection 
#%% step 5) Data preprocessing

vocab_size = 10000
oov_token ='<OOV>' 

## to learn
tokenizer= Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(articles) 
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# convert text to number
articles_int = tokenizer.texts_to_sequences(articles) 


#padding
max_len = np.median([len(articles_int[i]) for i in range (len(articles_int))])
padded_articles = pad_sequences(articles_int, maxlen= int(max_len), padding= 'post',
              truncating ='post'  )

# Y target 
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis = -1))


X_train,X_test, y_train,y_test = train_test_split(padded_articles, category,
                                                  test_size=0.3, 
                                                  random_state=123)


#%% model development
    
input_shape = np.shape(X_train)[1:]
out_dim = 32

model = Sequential()
model.add(Input(shape=(input_shape))) # LSTM,RNN, GRU ONLY ACCEPT 3D ARRAY
model.add(Embedding(vocab_size,out_dim))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(5,activation= 'softmax'))
model.summary()

plot_model(model,show_shapes=True,show_layer_names= True)

model.compile(optimizer= 'adam', 
              loss = 'categorical_crossentropy',
              metrics='acc')

#%% callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

#%% model training

hist= model.fit(X_train,y_train,
                epochs= 5, 
                callbacks = [tensorboard_callback],
                validation_data=(X_test,y_test))

#%% model evaluation 

y_pred = np.argmax(model.predict(X_test),axis = 1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual, y_pred))


#%% Evaluate the accuracy of our trained model
score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%model saving

#TOKENIZER
token_json =tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH, 'w') as file:
    json.dump(token_json,file)

#OHE
with open(OHE_SAVE_PATH, 'wb') as file:
    pickle.dump(ohe,file)
    
# Model
model.save(MODEL_SAVE_PATH)
