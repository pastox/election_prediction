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

from copy import deepcopy

##cross validation
def c_validation(X, y, k, function):
    kf = cross_validation.KFold(X.shape[0], n_folds=k)
    
    totalloss = 0 # Variable that will store the total intances that will be tested  
    totalsuccess5 = 0
    totalsuccess10 = 0	
    totalpercentageloss = 0
    res=[]
    corr=[]
    
    lines=[]
    
    for trainIndex, testIndex in kf:
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        avg = 0
        for i in trainLabels:
            avg+=i
        avg=avg/trainLabels.shape[0]
        
        predictedLabels = function(trainSet, trainLabels, testSet)
    
        loss = 0
        percentageloss=0
        success5 = 0
        success10 = 0	
        for i in range(testSet.shape[0]):
            if not np.isnan(predictedLabels[i][0]):
                if predictedLabels[i][0]>1:
                    predictedLabels[i][0]=avg
                loss+=abs(predictedLabels[i][0]-testLabels[i])/(testLabels[i]*testLabels.shape[0])
                percentageloss+=abs(predictedLabels[i][0]-testLabels[i])/testLabels.shape[0]
                if abs(predictedLabels[i][0]-testLabels[i])/testLabels[i] < 0.05:
                    success5+=1/testLabels.shape[0]
                    success10+=1/testLabels.shape[0]
                elif abs(predictedLabels[i][0]-testLabels[i])/testLabels[i] < 0.1:
                    success10+=1/testLabels.shape[0]
            else:
                print(i)
        print ('Loss: ', 100*loss, '%')
        print ('Average error: ', 100*percentageloss, '%')
        print ('Success 0.05: ',100*success5, '%')
        print ('Success 0.1: ',100*success10, '%')
        totalloss += loss
        totalpercentageloss += percentageloss
        totalsuccess5+=success5
        totalsuccess10+=success10
        
        res+=list(predictedLabels)
        corr+=list(testLabels[:])
        
    plt.plot(res, linestyle='', marker='.')
    plt.plot(corr, linestyle='', marker='.')    
    plt.show()
        
    print ('Total Loss: ',100*totalloss/k, '%')
    print ('Total Average Error: ',100*totalpercentageloss/k, '%')
    print ('Total success 0.05: ',100*totalsuccess5/k, '%')
    print ('Total success 0.1: ',100*totalsuccess10/k, '%')
    return totalloss/k
    #∟return lines

##data import

#data = pd.read_csv('C:\\Users\\Nicolas\\Desktop\\election_prediction\\macronData.csv')
data = np.genfromtxt('C:\\Users\\Nicolas\\Desktop\\election_prediction\\lepenData.csv', delimiter=',', skip_header = 1)
data=np.delete(data, 41, axis=1)
nan=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if np.isnan(data[i,j]):
            print(i,j)
            nan+=1
print(nan)

X = data[:,:-1]
y=data[:,-1]


##neural network

def neural_network(Xtrain,ytrain, Xtest):
    model = Sequential()
    model.add(Dense(30, activation='relu', input_dim=42))
    model.add(Dense(15, activation='relu', input_dim=42))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='SGD',
              loss='mean_absolute_error',
              metrics=['accuracy'])
    model.fit(Xtrain, ytrain, epochs=5, validation_split=0.2, verbose = 0)
    return model.predict(Xtest)

##
c_validation(X,y,5, neural_network)


"""for i in range(10):
    deb=1
    l=range(2100)
    for i in range(10):
        print(l)
        lines = c_validation(X,y,5, neural_network)
        aux=[]
        for e in lines:
            if e in l:
                aux.append(e)
        l=deepcopy(aux)
    print(l)"""
                
                
