import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


def classifier_parametertuning():
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit()

def crossvalidation(X_train_tfidf, cats_train):  
    clf = svm.SVC(kernel='linear', C=2)
    
    myscoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score),
               'recall' : make_scorer(recall_score), 
               'f1_score' : make_scorer(f1_score)}

    #scores = cross_val_score(clf, X_train_tfidf, cats_train, cv=5)
    #print(scores)
    results = cross_validate(clf, X_train_tfidf, cats_train, cv=5, scoring=myscoring)
    print(results)
    clf.fit( X_train_tfidf, cats_train)
    return clf
      
    
def prepare():
    # load data from file
    df_train = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/sms-train.csv')
    df_test = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/sms-evalAryan.csv')
    
    
    #balance the data so that train and test has the same size
    df_train= oversampling_me(df_train, 'HAM-SPAM') 
    
    # shuffle the  training data
    df_train = df_train.sample(frac=1).reset_index(drop=True)   
    
    
    # split dataframe into lists
    texts_train = df_train['SMS'].tolist()
    labels_train = df_train['HAM-SPAM'].tolist()
    
    #get the test text
    texts_test = df_test['SMS'].tolist()
    labels_test = df_test['HAM-SPAM'].tolist()
      
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocabulary
    X_train_tfidf=vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)
   
    #Convert the 'Yes' /'No' to binary 1/0 ofr the model processing
    cats_train = [ int(label == "Ham") for label in labels_train]
    cats_test = [ int(label == "Ham") for label in labels_test]
    #do cross validation with different subsets of 'test data' in each fold
    clf_trained=crossvalidation(X_train_tfidf,cats_train)
    
    predicted = clf_trained.predict(X_test_tfidf)
    
    print(accuracy_score(predicted, cats_test))
    
    
    #df_test["HAM-SPAM"]= predicted
    
    #df_test['HAM-SPAM'] = df_test['HAM-SPAM'].map({1: "Ham", 0: "Spam"})
    
    #df_test = df_test.drop(['SMS'], axis=1)
    
    #df_test.to_csv("answer_HamSpam.csv", index=False)
    


def oversampling_me(df_train,label_col):
    pos_sample = len(df_train[df_train[label_col]=='Ham'])
    neg_sample = len(df_train[df_train[label_col]=='Spam'])
    diff = pos_sample-neg_sample
    df_pos_samples = df_train[df_train[label_col]=='Ham']
    df_neg_samples = df_train[df_train[label_col]=='Spam']
    
    if diff > 0:
        df_oversample=df_neg_samples.sample(n=diff,random_state=42,replace=True)
    else:
        df_oversample=df_pos_samples.sample(abs(diff),random_state=42)
            
    df_train=df_train.append(df_oversample, ignore_index=True)
   
    print (len(df_train[df_train[label_col]=='Ham']))
    print (len(df_train[df_train[label_col]=='Spam']))
    #exit()
    df_train=shuffle(df_train)
    #out_bal=os.path.join(outBaseDir,"balanced_train.csv")
    #df_train.to_csv(out_bal,index=False)
    return df_train

if __name__ == '__main__':
    prepare()


    
    
    
    
