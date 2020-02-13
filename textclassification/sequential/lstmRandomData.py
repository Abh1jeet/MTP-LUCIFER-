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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import nltk


###train data
filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']
rows,columns=df.shape

train_text=[]
train_label=[]
for i in df.Text:
	train_text.append(i)
for i in df.Label:
	train_label.append(i)


###test data

filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['Text','Label']

test_text=[]
test_label=[]
for i in df.Text:
	test_text.append(i)
for i in df.Label:
	test_label.append(i)


total_text=test_text+train_text
total_label=test_label+train_label


def TFIDF(X_train,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(total_text).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train)





from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#text = ["The quick brown fox jumped over the lazy dog.",
		#"The dog.",
		#"The fox"]
#y=["c1","c2","c3"]

inputf= TFIDF(total_text)





encoder = LabelEncoder()
encoder.fit(total_label)
encoded_y = encoder.transform(total_label)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

#for i in dummy_y:
	#print(i)

#for i in inputf:
	#print(i)
input_dim=(inputf[0].shape)[0]

# X=[]
# for i in inputf:
# 	row=inputf[0].shape
# 	temp=[]	
# 	for j in i:
# 		temp.append(j)
# 	X.append(temp)

# print(inputf.shape[1])


# # define baseline model
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(16, input_dim=input_dim, activation='relu'))
	model.add(Dense(8, input_dim=input_dim, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=2, shuffle=True)
results = cross_val_score(estimator, inputf, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

