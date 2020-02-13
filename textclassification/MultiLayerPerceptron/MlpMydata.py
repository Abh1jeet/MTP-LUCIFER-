from __future__ import division, print_function
from sklearn.neural_network import MLPClassifier
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import collections
import re
import string

filePath="D:\\mtech4\\data\\combine.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
df.columns = ['TID', 'Text','Tag','Label']

text=[]
neu=[]         #0
neg=[]		   #1
pos=[]         #2
for i in df.Text:
	text.append(i)
for i in df.Label:
	if i==0:
		neu.append(1)
		neg.append(0)
		pos.append(0)
	if i==1:
		neu.append(0)
		neg.append(1)
		pos.append(0)
	if i==2:
		neu.append(0)
		neg.append(0)
		pos.append(1)

data=pd.DataFrame()
data['Text']=text
data['Pos']=pos
data['Neg']=neg
data['Neu']=neu
data['Label']=df['Label']

print(data.head())




#to remove panctuations from text

def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))




from nltk import word_tokenize, WordNetLemmatizer
tokens = [word_tokenize(sen) for sen in data.Text_Clean]


#converting text to lowercase
def lower_token(tokens): 
    return [w.lower() for w in tokens]    
lower_tokens = [lower_token(token) for token in tokens]



#removing stopwords
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
def remove_stop_words(tokens): 
    return [word for word in tokens if word not in stoplist]
filtered_words = [remove_stop_words(sen) for sen in lower_tokens]


result = [' '.join(sen) for sen in filtered_words]
data['Text_Final'] = result
data['tokens'] = filtered_words
#changing text to final text after removing panctuation and converting everything to lowercase
data = data[['Text_Final', 'tokens', 'Pos', 'Neg','Neu','Label']]


data_train, data_test = train_test_split(data, test_size=0.10, random_state=42)
all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
#print(data.head())

all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))


word2vec_path = 'D:\\mtech4\\embeddings\\googleNews300Negative\\GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)





def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)




training_embeddings = get_word2vec_embeddings(word2vec, data_train, generate_missing=True)

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300


tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))






train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)



train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)


test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)





label_names = ['Pos', 'Neg','Neu']
y_train = data_train[label_names].values
x_train = train_cnn_data
y_tr=[]
for i in y_train:
	if i[0]==1:
		y_tr.append(0)
	if i[1]==1:
		y_tr.append(1)
	if i[2]==1:
		y_tr.append(2)





text_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ), random_state=1)
text_clf.fit(x_train, y_tr)
predictions = text_clf.predict(test_cnn_data)

print(metrics.classification_report(data_test.Label, predictions))

r=0
for i in data_test.Label:
	print(i , predictions[r])
	r=r+1




























