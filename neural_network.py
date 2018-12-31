import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

##cross validation
def c_validation(X, y, k, function):
    kf = cross_validation.KFold(X.shape[0], n_folds=k)
    
    totalloss = 0 # Variable that will store the total intances that will be tested  
    totalsuccess5 = 0
    totalsuccess10 = 0	
    
    for trainIndex, testIndex in kf:
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        predictedLabels = function(trainSet, trainLabels, testSet)
    
        loss = 0
        success5 = 0
        success10 = 0	
        for i in range(testSet.shape[0]):
            if not np.isnan(predictedLabels[i][0]):
                loss+=abs(predictedLabels[i][0]-testLabels[i])/(testLabels.shape[0])
                if abs(predictedLabels[i][0]-testLabels[i])/testLabels[i] < 0.05:
                    success5+=1/testLabels.shape[0]
                elif abs(predictedLabels[i][0]-testLabels[i])/testLabels[i] < 0.1:
                    success10+=1/testLabels.shape[0]
        print ('Loss: ', loss)
        print ('Success 0.05: ',success5)
        print ('Success 0.1: ',success10)
        totalloss += loss
        totalsuccess5+=success5
        totalsuccess10+=success10
    print ('Total Loss: ',totalloss/k)
    print ('Total success 0.05: ',totalsuccess5/k)
    print ('Total success 0.1: ',totalsuccess10/k)
    return totalloss/k

##data import

#data = pd.read_csv('C:\\Users\\Nicolas\\Desktop\\election_prediction\\macronData.csv')
data = np.genfromtxt('C:\\Users\\Nicolas\\Desktop\\election_prediction\\macronData.csv', delimiter=',', skip_header = 1)

X = data[:,:-1]
y=data[:,-1]


##neural network
def get_fc_model():
    model = Sequential()
    #model.add(Dense(64, activation='relu', input_dim=43))
    model.add(Dense(32, activation='sigmoid', input_dim=43))
    model.add(Dense(16, activation='sigmoid', input_dim=43))
    model.add(Dense(8, activation='sigmoid', input_dim=43))
    model.add(Dense(4, activation='sigmoid', input_dim=43))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


def neural_network(Xtrain,ytrain, Xtest):
    fc = get_fc_model()
    fc.fit(Xtrain, ytrain, epochs=3, validation_split=0.2, verbose = 0)
    return fc.predict(Xtest)

##

c_validation(X,y,5, neural_network)
