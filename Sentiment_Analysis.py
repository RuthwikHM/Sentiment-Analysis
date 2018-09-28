#Importing the necessary modules
import keras
import numpy as np
import pandas as pd
from keras.layers import LSTM,Embedding,Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
#Reading the data
data=pd.read_csv('Sentiment.csv')
max_words=2000
#Preprocessing the data
tokenizer=Tokenizer(num_words=max_words,split=" ")
tokenizer.fit_on_texts(data['text'].values)
x_train=tokenizer.texts_to_sequences(data['text'].values)
x_train=sequence.pad_sequences(x_train)
y_train=pd.get_dummies(data['sentiment'])
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.3)
#Creating the model
model=Sequential()
model.add(Embedding(2000,128,input_length=29))
model.add(LSTM(3))
model.add(Dropout(0.2))
#Compiling the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Fiitng data to model
model.fit(x_train,y_train,epochs=20,batch_size=280,validation_split=0.3,verbose=2)
#Evaluating the model
score=model.evaluate(x_test,y_test,verbose=2,batch_size=280)
print('Evaluation accuracy=',(score[1]*100))
