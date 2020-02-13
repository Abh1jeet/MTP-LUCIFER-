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
filePath="D:\\mtech4\\cleaned_dataset.csv"

def TFIDF(X_train,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train)

df=pd.read_csv(filePath,header=None,skiprows=0,delimiter=",")
rows,columns=df.shape


#df[column][row]
#so column is 0 or 1 where 0 is question and 1 is tag


text=[]
y=[]
for i in range(1,rows):
	text.append(df[0][i])
	y.append(df[1][i])



from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#text = ["The quick brown fox jumped over the lazy dog.",
		#"The dog.",
		#"The fox"]
#y=["c1","c2","c3"]

inputf= TFIDF(text)





encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

for i in dummy_y:
	print(i)

for i in inputf:
	print(i)

print(inputf[0].shape)
input_dim=inputf[0].shape

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
	model.add(Dense(8, input_dim=466, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=2, shuffle=True)
results = cross_val_score(estimator, inputf, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

