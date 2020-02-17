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
filePath="D:\\mtech4\\data\\combine.txt"
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




def TFIDF(X_train,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(text).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train)





from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
#text = ["The quick brown fox jumped over the lazy dog.",
		#"The dog.",
		#"The fox"]
#y=["c1","c2","c3"]

inputf= TFIDF(text)



#give number to all the different classes
encoder = LabelEncoder()
encoder.fit(label)
encoded_y = encoder.transform(label)

#give one hot encoding to each label 
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

#for i in encoded_y:
	#print(i)

#for i in dummy_y:
	#print(i)

#for i in inputf:
	#print(i)

print(inputf[0].shape)
input_dim=(inputf[0].shape)[0]
print(input_dim) #inpurt_dim is the size of vocablury

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
	
	# ### model 1 with embedding layer
	# # create model
	# model = Sequential()
	# model.add(Embedding(75000,16))
	# model.add(GlobalAveragePooling1D())
	# model.add(Dense(16, input_dim=input_dim, activation='relu'))
	# model.add(Dense(3, activation='softmax'))
	# # Compile model
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	

	### model 1 with embedding layer and lstm layer
	model = Sequential()
	model.add(Embedding(input_dim, 100))### size of vocab ,embedding dimension , length of sentence
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	print(model.summary())
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=5, verbose=0)
kfold = KFold(n_splits=2, shuffle=True)
results = cross_val_score(estimator, inputf, dummy_y, cv=kfold)



print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




















