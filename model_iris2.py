from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
 
 
#===============================================================================
# data=load_iris()
#  
# X=data.data
# y=data.target
# 
# s=StandardScaler() #We are only transforming data not the labels
# 
# X1=s.fit_transform(X)
# X1_tr, X1_tst, y_tr, y_tst = train_test_split(X1,y, shuffle=True, random_state=32) #There are 4 features
# knn = KNeighborsClassifier(n_neighbors=5)
#  
# knn.fit(X1_tr, y_tr) 
# sc=knn.score(X1_tst, y_tst) 
# print('Score', sc)
# y_pred = knn.predict(X1_tst) 
# c = classification_report(y_tst, y_pred)
# print('Confusion Matrix', c )
#===============================================================================


# Load the bundled iris flower data set 
data = load_iris() 
X= data.data
y=data.target
# Try different normalizers and see the effect on final accuracy 
s=StandardScaler()

s=MinMaxScaler() #We are only transforming data not the labels 
X1=s.fit_transform(X) #X1=X
# shuffle the data and split 50-50

X_tr, X_tst, y_tr, y_tst = train_test_split(X1, y, shuffle=True, test_size=0.8, random_state=32)

pca = PCA(n_components=3) 

pca.fit(X_tr)

print('Feature Variance ratio ', pca.explained_variance_ratio_) 
X_tr_pca = pca.transform(X_tr) 
X_tst_pca = pca.transform(X_tst)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr_pca, y_tr) # accuracy compared to predicted labels 
score = knn.score(X_tst_pca, y_tst)
print("Accuracy ", score)

