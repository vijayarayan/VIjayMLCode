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
from gensim.models import word2vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import sys


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
    df_train = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/SRdataOneLineAscii.csv',low_memory=False, encoding='utf-8')
    #df_train = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/checkreduced.csv',low_memory=False, encoding='utf-8')
    df_train= df_train.dropna(subset=['Description'])
    df_train=df_train[df_train.Description != 'N/A']
    #df_train = df_train[df_train.Description.apply(lambda x: x.isnumeric())]
    #df_train = df_train[df_train.Description.apply(lambda x: x !="")]
    
    # add a new column in dataframe "label_col"
    df_train['label_col']=df_train.apply(func, axis=1)
    
    
    #balance the data so that train and test has the same size
    df_train= oversampling_me(df_train, 'label_col') 
    
    # shuffle the  training data
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    
    print( len(df_train)) 
    
    #Again take the small subset of data due to compute limitations
    #df_train = df_train.sample(n=10000, random_state=42)
    #print( len(df_train)) 
    
    # apply the clean function to df['text']
    df_train['Description'] = df_train['Description'].map(lambda x: clean_text(x))     
    
   
    # split dataframe into lists
    texts_train = df_train['Description'].tolist()
    labels_train = df_train['label_col'].tolist()
    
    #Tokenize and Create Sequence
    ### Create sequence
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words= vocabulary_size)
    tokenizer.fit_on_texts(df_train['Description'])
    sequences = tokenizer.texts_to_sequences(df_train['Description'])
    data = pad_sequences(sequences, maxlen=300)
    
    embeddings_index = dict()
    f = open('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/glove.6B/glove.6B.300d.txt', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
        
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector 
                
    ## Network architecture
    #===========================================================================
    # model = Sequential()
    # model.add(Embedding(20000, 300, input_length=10000))
    # model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #===========================================================================
    
    model_glove = Sequential()
    model_glove.add(Embedding(vocabulary_size, 300, input_length=300, weights=[embedding_matrix], trainable=True))
    model_glove.add(Dropout(0.2))
    model_glove.add(Conv1D(64, 5, activation='relu'))
    model_glove.add(MaxPooling1D(pool_size=2))
    model_glove.add(LSTM(300))
    model_glove.add(Dense(1, activation='sigmoid'))
    model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    ## Fit the model
    model_glove.fit(data, np.array(labels_train), validation_split=0.4, epochs=3, batch_size=500)      
    
    #texts_train = normalize_document(texts_train)
    # create the transform
    
    #vectorizer = TfidfVectorizer(stop_words='english')
    # tokenize and build vocabulary
    #X_train_tfidf=vectorizer.fit_transform(texts_train)
    
    #do cross validation with different subsets of 'test data' in each fold
    #clf_trained=crossvalidation(X_train_tfidf, labels_train)
    
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
    
    df_os = df_os.sample(n=5000, random_state=42)
    df_sysmgt = df_sysmgt.sample(n=5000, random_state=42)
    df_crash = df_crash.sample(n=5000, random_state=42) 
    df_net = df_net.sample(n=5000, random_state=42)
    df_storage = df_storage.sample(n=5000, random_state=42)
    df_install = df_install.sample(n=5000, random_state=42)
    df_other = df_other.sample(n=5000, random_state=42)
    
    df_train=pd.concat([df_os,df_sysmgt,df_crash,df_net,df_storage,df_install, df_other])
     
     
    print (len(df_train[df_train[label_col]== 0]))
    print (len(df_train[df_train[label_col]== 1]))
    print (len(df_train[df_train[label_col]== 2]))
    print (len(df_train[df_train[label_col]== 3]))
    print (len(df_train[df_train[label_col]== 4]))
    print (len(df_train[df_train[label_col]== 5]))
    print (len(df_train[df_train[label_col]== 6]))
    
    df_train=shuffle(df_train)
   
    return df_train  

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    
    return text



if __name__ == '__main__':
    prepare()


    
    
    
    
