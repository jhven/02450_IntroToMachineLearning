# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:40:55 2019

@author: Michael
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:33:51 2019

@author: sofierossen
"""

#Some code, is taken from the exercises in machinelearning 


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from scipy import stats
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.linalg import svd
from matplotlib.pyplot import (figure, subplot, plot, legend, show, hist,
                               xlabel, ylabel, xticks, yticks, boxplot, setp,title,ylim,subplot)
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

################################################
# OPENING AND READING THE FILE
################################################
#Open file 
attributeNames =["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
#C for correct,
attributeNamesC = ["Length","Diam","Height","Whole","Shell","Rings"]

df = pd.read_csv('../abalone.data',names=attributeNames)
#raw_data = df.get_values() 
df1 = df[df.columns[0:-1]]

################################################
# EXCLUSION OF OULIERS
################################################
df_noOutliers = df[df.Height < 0.4]
df_noOutliers = df_noOutliers[df_noOutliers.Height > 0]


################################################
# FEATURE EXTRACTION 
################################################

##### One-out-of-K coding ######
Infant = list()
Female = list()
Male = list()

for i in list(df_noOutliers["Sex"]):
    if i == "I":
        Infant.append(1)
    else:
        Infant.append(0)
    if i == "F":
        Female.append(1)
    else:
        Female.append(0)
    if i == "M":
        Male.append(1)
    else:
        Male.append(0)

rings = list(df_noOutliers["Rings"])
Age=list()
for i in range(len(rings)):
    Age = np.append(Age,rings[i]+1.5)
Newdf = pd.DataFrame({"Infant":Infant,"Female":Female,"Male":Male,"Length":list(df_noOutliers["Length"]),"Diam":list(df_noOutliers["Diam"]),"Height":list(df_noOutliers["Height"]),"Whole":list(df_noOutliers["Whole"]),"Shucked":list(df_noOutliers["Shucked"]),"Viscera":list(df_noOutliers["Viscera"]),"Shell":list(df_noOutliers["Shell"]),"Rings":list(df_noOutliers["Rings"]),"Age":Age}, dtype ="d")
attributeNames = list(Newdf.columns)

##### Standardizing the data #####
# Get column names first
names = Newdf.columns[3:]
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(Newdf.iloc[:, 3:])
scaled_df = pd.DataFrame(scaled_df, columns=names)
Newsclaeddf = pd.DataFrame({"Infant":Infant,"Female":Female,"Male":Male,"Length":list(scaled_df["Length"]),"Diam":list(scaled_df["Diam"]),"Height":list(scaled_df["Height"]),"Whole":list(scaled_df["Whole"]),"Shucked":list(scaled_df["Shucked"]),"Viscera":list(scaled_df["Viscera"]),"Shell":list(scaled_df["Shell"]),"Rings":list(scaled_df["Rings"]),"Age":list(scaled_df["Age"])}, dtype ="d")

################################################
# REGULARIZATION
################################################
## Defining x-matrix and y-vector and adding an offset attribute 
###################################################################################
###################################################################################
###################################################################################

#from main import *

#Transform the data into prober format 

#X = df_noOutliers[["Length","Diam","Height","Whole","Shell"]].get_values()
#y = df_noOutliers['Rings'].get_values()
#y.shape = [len(y),1]
#attributeNames = ["Length","Diam","Height","Whole","Shell"]
#N, M = X.shape

###################################################################################
###################################################################################
###################################################################################

X = np.concatenate((np.ones((Newsclaeddf.shape[0],1)),Newsclaeddf),1)
X = X[:,:-2]
y = Newsclaeddf["Age"].tolist() 
y=np.array(y,dtype=float)
attributeNames = [u'Offset']+attributeNames
M = len(attributeNames) -2
N=Newsclaeddf.shape[0]

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

## Values of lambda
lambdas = np.power(10.,range(-15,15))

## Initialize variables
T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))








