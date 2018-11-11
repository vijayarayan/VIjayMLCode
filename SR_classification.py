import pandas as pd
import re
import nltk
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline



def classifier_parametertuning():
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit()

def crossvalidation(X_train_tfidf, cats_train): 
    
    ch2 = SelectKBest(chi2, k=20000) 
    clf = svm.SVC(kernel='linear', C=2)
    ch2_svm = Pipeline([('chi', ch2), ('svc', clf)])
    # You can set the parameters using the names issued
    # For instance, fit using a k of 10 in the SelectKBest
    # and a parameter 'C' of the svm
    ch2_svm.set_params(chi__k=4000, svc__C=.1)
    #.fit(X_train_tfidf, cats_train)
    
    #ch2_svm.
    myscoring = {'accuracy' : make_scorer(accuracy_score), 
                'precision' : make_scorer(precision_score),
                'recall' : make_scorer(recall_score), 
                'f1_score' : make_scorer(f1_score)}
    
    results = cross_validate(ch2_svm, X_train_tfidf, cats_train, cv=3)
    print(results)
    clf.fit( X_train_tfidf, cats_train)
    return clf

def func(row):
    if row['GSS_Problem_Category__c'] == 'OS':
        return 0
    
    if row['GSS_Problem_Category__c'] == 'System Management':
        return 1
    if row['GSS_Problem_Category__c'] == 'Fault/Crash':
        return 2
          
    if row['GSS_Problem_Category__c'] == 'Networking':
        return 3
    
    if row['GSS_Problem_Category__c'] == 'Storage':
        return 4
    
    if row['GSS_Problem_Category__c'] == 'Installation':
        return 5
    
    return 6
      
        
    
    
def prepare():
    # load data from file
    df_train = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/SRdataOneLineAscii.csv',low_memory=False)
    
    df_train= df_train.dropna(subset=['Description'])
    df_train=df_train[df_train.Description != 'N/A']
    
    # add a new column in dataframe "label_col"
    df_train['label_col']=df_train.apply(func, axis=1)
    
    
    #balance the data so that train and test has the same size
    df_train= oversampling_me(df_train, 'label_col') 
    
    # shuffle the  training data
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    print( len(df_train)) 
    
    #Again take the small subset of data due to compute limitatitons
    df_train = df_train.sample(n=10000, random_state=42)
    print( len(df_train)) 
    
   
    # split dataframe into lists
    texts_train = df_train['Description'].tolist()
    labels_train = df_train['label_col'].tolist()
    
    #texts_train = normalize_document(texts_train)
    # create the transform
    
    vectorizer = TfidfVectorizer(stop_words='english')
    # tokenize and build vocabulary
    X_train_tfidf=vectorizer.fit_transform(texts_train)
    
    #do cross validation with different subsets of 'test data' in each fold
    clf_trained=crossvalidation(X_train_tfidf, labels_train)
    
    #predicted = clf_trained.predict(X_train_tfidf)
    
    #print(accuracy_score(predicted, labels_train))
       

def oversampling_me(df_train,label_col):
    len_cat_os = len(df_train[df_train[label_col]== 0])
    len_cat_sysmgt=len(df_train[df_train[label_col]== 1])
    len_cat_crash=len(df_train[df_train[label_col]== 2])
    len_cat_net=len(df_train[df_train[label_col]== 3])
    len_cat_storage=len(df_train[df_train[label_col]== 4])
    len_cat_install=len(df_train[df_train[label_col]== 5])
    len_cat_other=len(df_train[df_train[label_col]== 6])
    
    print(len_cat_os)
    print(len_cat_sysmgt)
    print(len_cat_crash)
    print(len_cat_net)
    print(len_cat_storage)
    print(len_cat_install)
    print(len_cat_other)
    
    df_os = df_train[df_train[label_col]== 0]
    df_sysmgt = df_train[df_train[label_col]== 1]
    df_crash = df_train[df_train[label_col]== 2]
    df_net = df_train[df_train[label_col]== 3]
    df_storage = df_train[df_train[label_col]== 4]
    df_install = df_train[df_train[label_col]== 5]
    df_other = df_train[df_train[label_col]== 6]
    
    df_os = df_os.sample(n=20000, random_state=42)
    df_sysmgt = df_sysmgt.sample(n=20000, random_state=42)
    df_crash = df_crash.sample(n=20000, random_state=42) 
    df_net = df_net.sample(n=20000, random_state=42)
    df_storage = df_storage.sample(n=20000, random_state=42)
    df_install = df_install.sample(n=20000, random_state=42)
    df_other = df_other.sample(n=20000, random_state=42)
    
    df_train=pd.concat([df_os,df_sysmgt,df_crash,df_net,df_storage,df_install, df_other])
     
    
    #===========================================================================
    # max_len_cat = max(len_cat_os,len_cat_sysmgt,len_cat_crash,len_cat_net, len_cat_storage,len_cat_install,len_cat_other )
    # 
    # if(max_len_cat > len_cat_os):
    #     diff_os = max_len_cat - len_cat_os
    # else:
    #     diff_os  = 0
    # 
    #     
    # if(max_len_cat > len_cat_sysmgt):
    #     diff_sysmgt = max_len_cat - len_cat_sysmgt
    # else:
    #     diff_sysmgt = 0
    #     
    # if(max_len_cat > len_cat_crash):
    #     diff_crash = max_len_cat - len_cat_crash
    # else:
    #     diff_crash = 0
    #     
    # if(max_len_cat > len_cat_net):
    #     diff_net = max_len_cat - len_cat_net
    # else:
    #     diff_net = 0
    #     
    #     
    # if(max_len_cat > len_cat_storage):
    #     diff_storage = max_len_cat - len_cat_storage
    # else:
    #     diff_storage = 0
    #    
    # 
    # if(max_len_cat > len_cat_install):
    #     diff_install = max_len_cat - len_cat_install
    # else:
    #     diff_install = 0
    #     
    # if(max_len_cat > len_cat_other):
    #     diff_other = max_len_cat - len_cat_other
    # else:
    #     diff_other = 0
    #            
    #     
    #     
    # if diff_os > 0:
    #     df_oversample_os=df_os.sample(n=diff_os,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_os, ignore_index=True)
    #     
    # if diff_sysmgt > 0:
    #     df_oversample_sysmgt=df_sysmgt.sample(n=diff_sysmgt,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_sysmgt, ignore_index=True)
    #     
    # if diff_crash > 0:
    #     df_oversample_crash=df_crash.sample(n=diff_crash,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_crash, ignore_index=True)
    #     
    # if diff_net > 0:
    #     df_oversample_net=df_net.sample(n=diff_net,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_net, ignore_index=True)
    #     
    # if diff_storage > 0:
    #     df_oversample_storage=df_storage.sample(n=diff_storage,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_storage, ignore_index=True)
    #     
    # if diff_install > 0:
    #     df_oversample_install=df_install.sample(n=diff_install,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_install, ignore_index=True)
    #     
    # if diff_other > 0:
    #     df_oversample_other=df_other.sample(n=diff_other,random_state=42,replace=True)
    #     df_train=df_train.append(df_oversample_other, ignore_index=True)
    #     
    #     
    #===========================================================================
    
    print (len(df_train[df_train[label_col]== 0]))
    print (len(df_train[df_train[label_col]== 1]))
    print (len(df_train[df_train[label_col]== 2]))
    print (len(df_train[df_train[label_col]== 3]))
    print (len(df_train[df_train[label_col]== 4]))
    print (len(df_train[df_train[label_col]== 5]))
    print (len(df_train[df_train[label_col]== 6]))
    
    df_train=shuffle(df_train)
   
    return df_train

wpt = nltk.WordPunctTokenizer()
#nltk.download()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(docs):
    # lower case and remove special characters\whitespaces
    #doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(docs)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
    

if __name__ == '__main__':
    prepare()


    
    
    
    
