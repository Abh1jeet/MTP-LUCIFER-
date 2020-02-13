from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk

filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']

#neu 0
#neg 1
#pos 2


text=[]
label=[]
for i in df.Text:
	text.append(i)
for i in df.Label:
	label.append(i)




filePath="D:\\mtech4\\data\\randomData\\combineSem16.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['Text','Label']

#neu 0
#neg 1
#pos 2


text_test=[]
label_test=[]
for i in df.Text:
	text_test.append(i)
for i in df.Label:
	label_test.append(i)



text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', tree.DecisionTreeClassifier()),
                     ])

text_clf.fit(text,label)


predicted = text_clf.predict(text_test)

print(metrics.classification_report(label_test, predicted))

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(label_test,predicted)
print('accuracy: {}'.format(accuracy))


