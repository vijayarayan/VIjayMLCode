'''
Created on Nov 15, 2016

@author: amita
'''
import os
import numpy as np
import random
seed=100
np.random.seed(seed)
random.seed(seed)
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Convolution1D
from keras.layers import Merge
import sys,io
import pandas as pd
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
import time
from src import file_utilities

history = History()
GLOVE_DIR = "Users/vmisra/eclipse-workspace/MLProject/data/vmdata/glove.6B/"
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
TRUNC='pre'
PADDING="post"
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

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

def vectorize_text (input_train,input_test,textcolumn,features,labelcol):
    """ vectorize the text samples into a 2D integer tensor
    """
    global word_index, train_length
    df_train=pd.read_csv(input_train)
    train_texts=df_train[textcolumn].values
    labels_train=df_train[labelcol].values
    
    readabilty_features_train = df_train[features]
#    print readabilty_features_train
    readabilty_features_train = np.reshape(np.ravel(np.ravel(readabilty_features_train)), (len(readabilty_features_train), len(features), 1))
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
    labels_test=df_test[labelcol].values

    readabilty_features_test = df_test[features]
    readabilty_features_test = np.reshape(np.ravel(np.ravel(readabilty_features_test)), (len(readabilty_features_test), len(features), 1))

    sequences_test = tokenizer.texts_to_sequences(test_texts)
    data_test= pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,truncating=TRUNC,padding=PADDING)
    labels_test = to_categorical(encoder.transform(labels_test))
    train_length = data_train.shape[0]
    print('Shape of data tensor:', data_train.shape)
    print('Shape of label tensor:', labels_train.shape)
    return(data_train,labels_train,data_test,labels_test,readabilty_features_train,readabilty_features_test)


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

def split_train_val(data,labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return(x_train,y_train,x_val,y_val)
    

def create_model(dropout,dim,optimizer='rmsprop'):

    global train_length, readabilty_features_count
    model = Sequential()
    embedding_layer= create_embedding_matrix(word_index)
    model.add(embedding_layer)
    model.add(Dropout(dropout)) 

    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(dropout)) 
    model.add(Bidirectional(LSTM(dim)))

#    model_readabilty = Sequential()
#    model_readabilty.add(Bidirectional(LSTM(readabilty_features_count), input_shape=(readabilty_features_count, 1)))
    
#   Merge

#    merged = Merge([model, model_readabilty], mode='concat')
#    model_merged = Sequential()
#    model_merged.add(merged)
#    model_merged.add(Dropout(0.1))
#    model_merged.add(Dense(2, activation='sigmoid'))
    
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
    return model
    
def writeresultstoFile(csvFile,pred_list,Y_pred,Y_actual,input_test,f1weighted):
    df_test=pd.read_csv(input_test)
    assert(len(Y_pred)==len(Y_actual)==df_test.shape[0])
    type(Y_pred)
    df_test["Y_actual"]=Y_actual
    df_test["Y_predicted"]=Y_pred
    df_test.to_csv(csvFile,index=False)
    f = io.open(csvFile, "a",encoding='utf-8')
    result= str("\nf1weighted") +str(f1weighted)
    pprint(result, stream=f) 
#     print("classification report {0}".format(classification_report(Y_pred,Y_actual)))
#     print("confusion_matrix() {0}".format(confusion_matrix(Y_pred,Y_actual)))
#     print("fscore {0}".format(f1weighted))
    f.close()


def run(outBaseDir,input_train,input_test,text_col,features,label_col,resfile,dropoutlist,dimlist,outputfile, req_features): 
    data_train,labels_train, data_test,labels_test,readabilty_features_train,readabilty_features_test=vectorize_text(input_train,input_test,text_col,features,label_col) 
    x_train,y_train= data_train,labels_train
    x_test, y_test= data_test,labels_test
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [history, checkpoint]
# Fit the model
    for droput in dropoutlist:
        for dim in dimlist:
            model=create_model(droput,dim)
            model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=64, callbacks=callbacks_list, verbose=0)
            scores = model.evaluate(x_test, y_test, verbose=0)
            with open(outputfile, 'a') as out:
                pprint("features {0}".format(req_features),stream=out)
                pprint("droput is {0} and dim is {1} and random is {2}\n".format(droput,dim,seed),stream=out)
                pprint("Accuracy: %.2f%% \n" % (scores[1]*100),stream=out)
            #    pred = model.predict([x_test, readabilty_features_test], batch_size=64)
                pred = model.predict(x_test, batch_size=64)
                y_pred_values=pred.argmax(axis=1)
                y_test_values=y_test.argmax(axis=1)
                pprint ("F-score: {0} ".format(f1_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted')),stream=out)
                pprint ("Confusion matrix is {0}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))),stream=out)
                filename="lstm"+text_col+"drop"+str(droput)+"dim"+str(dim)+".csv"
                csvwithpath=os.path.join(outBaseDir,filename)
                f1weighted=f1_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted')
                pred_list=[{'Y_actual': v1, 'Y_predicted': v2} for v1, v2 in zip( y_test_values,y_pred_values)]
                writeresultstoFile(csvwithpath,pred_list,y_pred_values,y_test_values,input_test,f1weighted)
 
       
if __name__ == '__main__':

    global readabilty_features_count
    prefix = os.getcwd()
    prefix = file_utilities.get_absolutepath_data("data/vmdata")
    input_train= prefix + "/SRdataOneLineAscii.csv"
    input_test= prefix + "/all_mtbatch_GCTest_Coref_featuresbalance.csv"
    outputfile=prefix+"/GC_LSTM_onlysentence_results2.txt"
    dropoutlist=[0.2]
    dimlist=[200]
    text_col="sent"
    label_col="class"
    
    df_train = pd.read_csv(input_train)
    all_features = list(df_train)
#    All features that start with any of the below names
    req_features = []
    features = []
    
    for feature in all_features:
        for prefix in req_features:
            if feature.startswith(prefix):
                features.append(feature)
    readabilty_features_count = len(features)
    resfile=prefix+"/GC_Cleanbalance_progress"
    print (features)
    run(prefix,input_train,input_test,text_col,features,label_col,resfile,dropoutlist,dimlist,outputfile, req_features)
