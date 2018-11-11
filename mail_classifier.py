
from __future__ import print_function

import os,io
import sys
import numpy as np
import pandas as pd
import random
seed=100
np.random.seed(seed)
random.seed(seed)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import codecs
from pprint import pprint
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import keras.backend as K
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Convolution1D
from src import file_utilities
from keras.callbacks import History
from keras.optimizers import Adam
history = History()
import time
TRUNC='pre'
PADDING="post"

BASE_DIR = ''
#GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
GLOVE_DIR = '/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/glove.6B/'
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# first, build index mapping words in the embeddings set
# to their embedding vector
from keras.callbacks import ModelCheckpoint
word_index={}
embeddings_index = {}
train_length = 0
readabilty_features_count = 0
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), "r", encoding='utf-8')
#f = open('glove.6B.100d.txt', 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

def preprocess_text(texts):
    return [x.lower() for x in texts]
    
def upsampledf(df_train,labelcol):

    #(pos_sample + neg_sample) / (2 * pos_sample)
    ham_sample = len(df_train[df_train[labelcol]=="Ham"])
    spam_sample = len(df_train[df_train[labelcol]=="Spam"])
    print(ham_sample)
    print(spam_sample)
    ratio = np.round((ham_sample + spam_sample) / (2 * ham_sample))

    print ("ratio" , ratio)

    df_ham_samples = df_train[df_train[labelcol]=="Ham"]
    df_spam_samples = df_train[df_train[labelcol]=="Spam"]

    print (len(df_train[df_train[labelcol]=="Ham"]))
    print (len(df_train[df_train[labelcol]=="Spam"]))

    df_spam_oversamples = df_spam_samples
    for i in range(int(ratio)+1):
        df_spam_oversamples = df_spam_oversamples.append(df_spam_samples)

    df_train = df_ham_samples.append(df_spam_oversamples)

    print (len(df_train[df_train[labelcol]=="Ham"]))
    print (len(df_train[df_train[labelcol]=="Spam"]))
    #exit()
    return df_train

def vectorize_text (input_train,input_test,textcolumn,labelcol):
    """ vectorize the text samples into a 2D integer tensor
    """
    global word_index, train_length
    df_train=pd.read_csv(input_train)
    df_train = upsampledf(df_train, "HAM-SPAM")
    train_texts=df_train[textcolumn].values
    train_texts=preprocess_text(train_texts)
    labels_train=df_train[labelcol].values
    
#    print readabilty_features_train
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_texts)
    sequences_train = tokenizer.texts_to_sequences(train_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels_train)
    encoded_labels_train = encoder.transform(labels_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels_train= to_categorical(encoded_labels_train)
    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH,truncating=TRUNC,padding=PADDING)
    
    df_test=pd.read_csv(input_test)
    test_texts=df_test[textcolumn].values
    test_texts=preprocess_text(test_texts)
    #labels_test=df_test[labelcol].values
    sequences_test = tokenizer.texts_to_sequences(test_texts)
    data_test= pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,truncating=TRUNC,padding=PADDING)
    #labels_test = to_categorical(encoder.transform(labels_test))
    
    
    train_length = data_train.shape[0]
    
    print('Shape of data tensor:', data_train.shape)
    print('Shape of label tensor:', labels_train.shape)
    #return(data_train,labels_train,data_test,labels_test)
    return(data_train,labels_train,data_test)

def create_embedding_matrix(word_index):    
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    return embedding_layer

def writeresultstoFile(csvFile,pred_list,Y_pred,input_test):
    df_test=pd.read_csv(input_test)
    #assert(len(Y_pred)==len(Y_actual)==df_test.shape[0])
    type(Y_pred)
    #df_test["Y_actual"]=Y_actual
    df_test["Y_predicted"]=Y_pred
    df_test.to_csv(csvFile,index=False)
    f = io.open(csvFile, "a",encoding='utf-8')
    #result= str("\nf1weighted") +str(f1weighted)
    #pprint(result, stream=f) 
    f.close()

def create_model(dropout,dim,optimizer='rmsprop'):

    global train_length, readabilty_features_count
    model = Sequential()
    embedding_layer= create_embedding_matrix(word_index)
    model.add(embedding_layer)
    model.add(Dropout(dropout)) 

    model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(dropout)) 
    model.add(Bidirectional(LSTM(dim)))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
    return model
    


def run(outBaseDir,input_train,input_test,text_col,label_col,resfile,dropoutlist,dimlist,outputfile): 
    data_train,labels_train, data_test =vectorize_text(input_train,input_test,text_col,label_col) 
    x_train,y_train= data_train,labels_train
    x_test = data_test
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [history, checkpoint]
# Fit the model
    for droput in dropoutlist:
        for dim in dimlist:
            model=create_model(droput,dim)
            model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32, callbacks=callbacks_list, verbose=0)
            #scores = model.evaluate(x_test, y_test, verbose=0)
            with open(outputfile, 'a') as out:
                pprint("droput is {0} and dim is {1} and random is {2}\n".format(droput,dim,seed),stream=out)
                #pprint("Accuracy: %.2f%% \n" % (scores[1]*100),stream=out)
            #    pred = model.predict([x_test, readabilty_features_test], batch_size=64)
                pred = model.predict(x_test, batch_size=64)
                y_pred_values=pred.argmax(axis=1)
                #y_test_values=y_test.argmax(axis=1)
                #pprint ("F-score: {0} ".format(f1_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted')),stream=out)
                #pprint ("Confusion matrix is {0}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))),stream=out)
                filename="lstm"+text_col+"drop"+str(droput)+"dim"+str(dim)+".csv"
                csvwithpath=os.path.join(outBaseDir,filename)
                #f1weighted=f1_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted')
                #pred_list=[{'Y_actual': v1, 'Y_predicted': v2} for v1, v2 in zip( y_test_values,y_pred_values)]
                pred_list=[{'Y_predicted': v2} for  v2 in y_pred_values]
                writeresultstoFile(csvwithpath,pred_list,y_pred_values,input_test)
 
       
if __name__ == '__main__':

    prefix = os.getcwd()
    prefix = file_utilities.get_absolutepath_data("vmdata")
    input_train= prefix + "/sms-train.csv"
    input_test= prefix + "/sms-eval.csv"
    outputfile=prefix+ "/ham_spamresults.txt"
    dropoutlist=[0.2]
    dimlist=[100]
    text_col="SMS"
    label_col="HAM-SPAM"
    
    resfile=prefix+"/progress"
    run(prefix,input_train,input_test,text_col,label_col,resfile,dropoutlist,dimlist,outputfile)




