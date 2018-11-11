
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

col = ['c1', 'c2', 'c3','c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'target']
data = pd.read_csv('c:\\workspace\\data.csv', header=None, names=col, index_col=col[0])
data = data.replace(to_replace='?', value=0)
X = data[col[1:10]].values
X = np.asarray(X, dtype='float64')
y = data['target']. values y = np.asarray(y, dtype='int32')
oE = OneHotEncoder(sparse=False) oE.fit(y.reshape(1,-1))
y=oE.fit_transform(y.reshape(y.shape[0],1))[:,0]
#y=oE.transform(y.reshape(1,-1))

s=StandardScaler()
#We are only transforming data not the labels
X1=s.fit_transform(X)
X1_tr, X1_tst, y_tr, y_tst = train_test_split(X1,y, shuffle=True, random_state=32, test_size=0.5)
y_tst_unlabel = np.full(y_tst.shape, -1, dtype='int32')
X1_new = np.concatenate((X1_tr, X1_tst), axis=0)
y_new = np.concatenate((y_tr,y_tst_unlabel), axis=0)

#Label propogation bug, make n_neighbors greater than default=9 #https://github.com/scikit-learn/scikit-learn/issues/9292 # Please comment or uncomment following APi as applicable #sms = LabelPropagation(kernel='knn', n_neighbors=11, max_iter=100000)
sms = LabelSpreading(kernel='knn', n_neighbors=11, max_iter=10000)
sms.fit(X1_new,y_new)
y_derived = sms.transduction_
X1_tr, X1_tst, y_tr, y_tst = train_test_split(X1_new,y_derived, shuffle=False, random_state=32, test_size=0.8)

#There are 4 features
knn = KNeighborsClassifier(n_neighbors=11) knn.fit(X1_tr, y_tr) sc=knn.score(X1_tst, y_tst)
print('Score ', sc)
y_pred = knn.predict(X1_tst)
c = classification_report(y_tst, y_pred)
print(' Confusion Matrix ', c )