import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def prepare():
    # load data from file
    df = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/tweets-train.csv')
    df_test = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/tweets-eval.csv')
    
    # shuffle the  training data
    df = df.sample(frac=1).reset_index(drop=True)
    
    #balance the data so that train and test has the same size
    df= oversampling_me(df, 'EXISTENCE')
    # split dataframe into lists
    texts = df['TWEET'].tolist()
    labels = df['EXISTENCE'].tolist()
    
    texts_test = df_test['TWEET'].tolist()
    #labels_test = df_test['EXISTENCE'].tolist()
    
   
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocabulary
    X_train_tfidf=vectorizer.fit_transform(texts)
    X_test_tfidf = vectorizer.transform(texts_test)
    # summarize
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    
    cats = [ int(label == "Yes") for label in labels]
    #cats_test = [ int(label == 'Yes') for label in labels_test]
    print(cats)
    
    clf = MultinomialNB().fit(X_train_tfidf, cats)
    
    predicted = clf.predict(X_test)
    #np.mean(predicted == cats_test)
    
    




def oversampling_me(df_train,label_col):
    pos_sample = len(df_train[df_train[label_col]=='Yes'])
    neg_sample = len(df_train[df_train[label_col]=='No'])
    diff = pos_sample-neg_sample
    df_pos_samples = df_train[df_train[label_col]=='Yes']
    df_neg_samples = df_train[df_train[label_col]=='No']
    
    if diff > 0:
        df_oversample=df_neg_samples.sample(n=diff,random_state=42,replace=True)
    else:
        df_oversample=df_pos_samples.sample(abs(diff),random_state=42)
            
    df_train=df_train.append(df_oversample, ignore_index=True)
   
    print (len(df_train[df_train[label_col]=='Yes']))
    print (len(df_train[df_train[label_col]=='No']))
    #exit()
    df_train=shuffle(df_train)
    #out_bal=os.path.join(outBaseDir,"balanced_train.csv")
    #df_train.to_csv(out_bal,index=False)
    return df_train

if __name__ == '__main__':
    prepare()


    
    
    
    
