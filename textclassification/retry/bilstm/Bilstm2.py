import pandas as pd 
import numpy as np 
import re
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation ,Bidirectional
from keras.layers.embeddings import Embedding

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import nltk
filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']
rows,columns=df.shape
#neu 0
#neg 1
#pos 2


text=[]
label=[]
for i in df.Text:
	text.append(i)
for i in df.Label:
	label.append(i)


########## text  preprocessing ######################################### 

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(text)

#pad sequences
max_length=max([len(s.split()) for s in text])

#define vocablury size
vocab_size=len(tokenizer_obj.word_index)+1


print(max_length)
print(vocab_size)



##generating tokens
text_token=tokenizer_obj.texts_to_sequences(text)

##adding padding 
text_token_withpad=pad_sequences(text_token,maxlen=max_length,padding='post')



#give number to all the different classes
encoder = LabelEncoder()
encoder.fit(label)
encoded_y = encoder.transform(label)

#give one hot encoding to each label 
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)








#######building model#########
EMBEDDING_DIM=100

# main model
model = Sequential()
model.add(Embedding(vocab_size,EMBEDDING_DIM,input_length=max_length))### size of vocab ,embedding dimension , length of sentence
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())



X_train, X_test, y_train, y_test = train_test_split(text_token_withpad, dummy_y, test_size=0.33, random_state=42)

#training
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=5, batch_size=5)

#testing
y_pred = model.predict(X_test)
for i in y_pred:
	print(i)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

for i in range(0,len(test)):
	print(pred[i] , test[i])

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)