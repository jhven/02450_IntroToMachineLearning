# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:07:50 2019

@author: Michael
"""

from main import * 

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import scipy.stats as st


#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

#Transform the data into prober format 

# Add binary attribute of whether abalone is adult (i.e. female or male) or infant
df_noOutliers['Adult'] = df_noOutliers.Sex != 'I'
df_noOutliers.Adult = df_noOutliers.Adult.astype(int)

# Extract vector y, convert to NumPy array
y = df_noOutliers.Adult.squeeze().to_numpy()

# Creating matrix X, only for the attributes of interest
X = df_noOutliers[attributeNamesC].to_numpy()

# Computing M, N and C
N, M = X.shape
C = len(df_noOutliers.Adult.unique())

# Define training set and test set from the dataframe
max_training_index = 3000


## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10 #Outer cross validation 
K2 = 10 #Inner cross validation 

K = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)


######################################################
### K-nearest neighbors
print("\n\n --- Classification with method: K-nearest neighbours\n")
dist=2
metric = 'minkowski'
metric_params = {}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
########################################################
lambda_interval = np.logspace(-8, 2, K)

#Error list 
Error_Cluster = list()
Error_Logistic = list()
Error_Baseline = list()
Error_Cluster_K = list()

k1 = 0 
#Outer loop 
for train_index, test_index in CV1.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    ###########################
    #    Initialization       #
    ###########################    
    Error_Cluster_Inner = list()
    Error_Logistic_Inner = list()
    Error_Baseline_Inner = list()
    Error_Basline_Picked = list()
    
        
    # Inner Loop 
    KK = 5
    for train_index2, test_index2 in CV2.split(X_train):
        
        #Extract new training set, of the current one. 
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2] 
        
        #############################
        #     K-nearest neighbors   #
        #############################
        
        #Change K  
        knclassifier = KNeighborsClassifier(n_neighbors=KK, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
        KK +=1
        
        knclassifier.fit(X_train2, y_train2)
        y_est = knclassifier.predict(X_test2)

        Error_Cluster_Inner.append(np.sum(y_est!=y_test2)/np.sum(y_test2))
           
        
        ###########################
        #   Logistisk regression #
        ###########################
        logisticClassifier = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[k1])
    
        logisticClassifier.fit(X_train2, y_train2)

        y_train_est = logisticClassifier.predict(X_train2).T
        y_test_est = logisticClassifier.predict(X_test2).T
    
        #train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train2)
        Error_Logistic_Inner.append(np.sum(y_test_est != y_test2) / len(y_test2))

        #w_est = logisticClassifier.coef_[0] 
        #coefficient_norm[k] = np.sqrt(np.sum(w_est**2))


        ###########################
        #     BASELINE            #
        ###########################
        
        # IS THIS TECHINICAL THE BEST WAY? 
        unique_baseline, counts_baseline = np.unique(y_train2, return_counts=True)
        baseline_count_dict = dict(zip(unique_baseline, counts_baseline))
        Error_Baseline_Inner.append(1-sum(y_test2 == 1)/len(y_test2))
        
        #Error_Basline_Picked.append()
        
    #################################################################
    # Test model on full set 
    ######################################################################

    ###########################
    #      K-nearest neighbors   #
    ###########################        
    #Pick best model 
    best_K = np.argmin(Error_Cluster_Inner)
    Error_Cluster_K.append(int(best_K)+5)
    # +1 Due to python counts from zero with position 
    knclassifier = KNeighborsClassifier(n_neighbors=(best_K), p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
    
    #New test
    knclassifier.fit(X_train, y_train)
    y_est = knclassifier.predict(X_test)

    Error_Cluster.append(np.sum(y_est!=y_test)/np.sum(y_test))   
    
    
    ###########################
    #   Logistisk regression #
    ###########################
    #Pick best model
    best_lambda = np.argmin(Error_Logistic_Inner)
    
    logisticClassifier = lm.LogisticRegression(penalty='l2', C=1/best_lambda)

    #New test
    logisticClassifier.fit(X_train, y_train)
    y_test_est = logisticClassifier.predict(X_test).T

    Error_Logistic.append(np.sum(y_test_est != y_test) / len(y_test))



    ###########################
    #     BASELINE            #
    ###########################    
    
    #THIS IS TECHNICAL NOT CORRECT!
    unique_baseline, counts_baseline = np.unique(y_train, return_counts=True)
    baseline_count_dict = dict(zip(unique_baseline, counts_baseline))
    Error_Baseline.append((1-sum(y_test == 1)/len(y_test)))
    

    print('Cross validation fold {0}/{1}'.format(k1+1,K))
    k1 +=1



#Output to Latex
table =[ Error_Cluster_K, Error_Cluster, Error_Logistic, Error_Baseline ]
table = np.array(table).T.tolist()
#print(tabulate(table))
print(tabulate(table, headers=["k_cluster", "Cluster", "Logistic", "Baseline"],tablefmt="latex"))    
