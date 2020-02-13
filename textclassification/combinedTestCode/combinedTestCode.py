from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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




filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"
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
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('Decision Tree accuracy: {}'.format(accuracy))




from sklearn.ensemble import GradientBoostingClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', GradientBoostingClassifier(n_estimators=100)),
                     ])

text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('Gradient GradientBoostingClassifier: {}'.format(accuracy))


from sklearn.neighbors import KNeighborsClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('KNeighborsClassifier: {}'.format(accuracy))



from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BaggingClassifier(KNeighborsClassifier())),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('KNeighborsClassifier(Bagging): {}'.format(accuracy))


from sklearn.svm import LinearSVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('svm(Linear): {}'.format(accuracy))






from sklearn.svm import SVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(kernel='poly',degree=8)),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('svm(Poly): {}'.format(accuracy))

from sklearn.svm import SVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(kernel='rbf')),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('svm(rbf): {}'.format(accuracy))



from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('MultinomialNB: {}'.format(accuracy))


from sklearn.neighbors.nearest_centroid import NearestCentroid
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', NearestCentroid()),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('NearestCentroid: {}'.format(accuracy))


from sklearn.ensemble import RandomForestClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])
text_clf.fit(text,label)
predicted = text_clf.predict(text_test)
#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(label_test,predicted)
print('RandomForestClassifier: {}'.format(accuracy))




