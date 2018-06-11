
from keras.models import Sequential
from keras.layers import Flatten,Dense, Dropout,Conv2D, Activation, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,SGD
import keras.backend as K
import matplotlib.image as mpimg
import numpy as np
import scipy.io
import tensorflow as tf 
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from keras.optimizers import Adam,SGD

# The method that creates the model
def make_model(load=False, filepath='./Saved_model', optim=None):
    #Preventing none parameter
    if(optim == None):
        optim = 'adam'
    model = Sequential()
    model.add(Convolution1D(
        filters=32, 
        activation='relu',
        input_shape=(15, 1),
        kernel_size=2, kernel_initializer="uniform"))
    model.add(Convolution1D(
        filters=64, 
        input_shape=(32, 1),
        activation='relu',
        kernel_size=2, kernel_initializer="uniform"))
    model.add(Convolution1D(
        filters=124,
        input_shape=(64, 1),
        activation='relu',
        kernel_size=2, kernel_initializer="uniform"))
    model.add(Convolution1D(
        filters=32,
        input_shape=(124, 1),
        activation='relu',
        kernel_size=2, kernel_initializer="uniform"))


    model.add(Flatten())
    model.add(Dense(units=512,  activation='softmax'))
    model.add(Dense(units=15,  activation='softmax'))

    if(load):
        model.load_weights(filepath)

    model.compile(optimizer=optim,
                  loss=custom_loss)
    return model

# A custom K mean loss function for the model
def custom_loss(y_true, y_pred): 
    error = K.mean(K.abs(y_pred - y_true ))
    return error

# Produces the sentences of length 15 that the model accepts 
def testing(Discharge_Summary):
    common_diseases = np.load("common_diseases.npy")
    # Extract all sentences from the Discharge summaries
    rgx = re.compile("([\w][\w']*\w)")
    strings = list()
    for i in Discharge_Summary:

        if(str(i) == "nan"):
            continue
        strings+= re.split("\.\s*",i.lower())
    all_sentences = list(strings)
    for ind in range(len(all_sentences)):

        all_sentences[ind] = rgx.findall(all_sentences[ind])
        

    training_sentences = list()
    for sentence in all_sentences:
        while(len(sentence) != 0):
            if(len(sentence) < 15):
                sentence+=list(np.zeros(15-len(sentence)))
            else:
                training_sentences += [sentence[:15]]
                sentence = sentence[15:]

    return training_sentences

# Transform the sentences of length 15 words to vectors of integers each, integer representing a unique word
def num_p(training_sentences,dic):
    num_representation = list()
    for sentence in training_sentences:
        temp_list = list()
        for word in sentence:
            if word in dic:
                temp_list += [dic[str(word)]]
            else:
                temp_list += [0]
        num_representation+=[temp_list]
    return num_representation
      
    

# Predict method either accepts an integer number which is presenting the index for the discharge summary in the
# NoTEEVENTS.csv. In the case that NOTEEVENTS.csv file is not available or that it takes a long time to load the entire
# File particiapnts can provide the discharge summary itself and the methods will feed it to the model in order to predict
# the diseases.
def predict(last=0,discharge=0):
    
    if(last!=0):
        pandaV = pd.read_csv("NOTEEVENTS.csv",low_memory=False)
        NoteEvents_Data= np.array(pandaV)
        Discharge_Summary = NoteEvents_Data[str(NoteEvents_Data[:,6]) == "Discharge summary",10][last-1:last]
    if(discharge!=0):
        Discharge_Summary=discharge

    words = np.load("words.npy")
    dic = {word: i+1 for i,word in enumerate(words)}
    dic['0.0']=0

    training_sentences = testing(Discharge_Summary)

    num_representation = num_p(training_sentences,dic)
    

    adam = Adam(lr=.0001, decay=.0)
    model = make_model(load=True,filepath='model_weights.h10',optim=adam)

    a = model.predict(np.array(num_representation).reshape(np.array(num_representation).shape + (1,)), batch_size=1)
    b = training_sentences

    
    diseases = list()
    for i,sentence in enumerate(a):
        if 1 in list(sentence>.999):
            ind = list(sentence>.999).index(1)
            diseases+=[b[i][ind]]
    return a, b, diseases