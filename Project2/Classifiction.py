# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:07:50 2019

@author: Michael
"""

from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import scipy.stats as st
import operator


#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
Error_Logistic_Lambda = list()
Error_Baseline = list()
Error_Baseline_Picked = list()
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
    Error_Logistic_Lambda_Inner = list()
    Error_Baseline_Inner = list()
    Error_Baseline_Picked_Inner = list()
    
        
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
        #   Logistisk regression  #
        ###########################
        best_lambda_inner = 1000.0
        current_best_logistic_error = 999999.99 #Initial value to guarantee update on first iteration
        for l in range(len(lambda_interval)):
            logisticClassifier = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[l])
        
            #Test the model
            logisticClassifier.fit(X_train2, y_train2)
            y_test_est = logisticClassifier.predict(X_test2)
            current_error = np.sum(y_test_est != y_test2) / len(y_test2)
            
            #Check if current model is better than the previous best inner model
            if current_error <= current_best_logistic_error:
                best_lambda_inner = lambda_interval[l]
                current_best_logistic_error = current_error

        Error_Logistic_Inner.append(current_best_logistic_error)
        Error_Logistic_Lambda_Inner.append(best_lambda_inner)


        ###########################
        #     BASELINE            #
        ###########################
        
        unique_baseline, counts_baseline = np.unique(y_train2, return_counts=True)
        baseline_count_dict = dict(zip(unique_baseline, counts_baseline))
        majority_value = max(baseline_count_dict.items(), key=operator.itemgetter(1))[0]
        Error_Baseline_Inner.append(1-sum(y_test2 == int(majority_value))/len(y_test2))
        Error_Baseline_Picked_Inner.append(int(majority_value))
        
        
    #################################################################
    # Test model on full set 
    ######################################################################

    ##############################
    #      K-nearest neighbors   #
    ##############################      
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
    
    logisticClassifier = lm.LogisticRegression(penalty='l2', C=1/Error_Logistic_Inner[best_lambda])

    #New test
    logisticClassifier.fit(X_train, y_train)
    y_test_est = logisticClassifier.predict(X_test).T

    Error_Logistic.append(np.sum(y_test_est != y_test) / len(y_test))
    Error_Logistic_Lambda.append(Error_Logistic_Inner[best_lambda])



    ###########################
    #     BASELINE            #
    ###########################    
    
    unique_baseline, counts_baseline = np.unique(y_train, return_counts=True)
    baseline_count_dict = dict(zip(unique_baseline, counts_baseline))
    majority_value = max(baseline_count_dict.items(), key=operator.itemgetter(1))[0]
    Error_Baseline.append((1-sum(y_test == int(majority_value))/len(y_test)))
    Error_Baseline_Picked.append(int(majority_value))
    

    print('Cross validation fold {0}/{1}'.format(k1+1,K))
    k1 +=1



#Output to Latex
table =[ Error_Cluster_K, Error_Cluster, Error_Logistic_Lambda, Error_Logistic, Error_Baseline_Picked, Error_Baseline ]
table = np.array(table).T.tolist()
#print(tabulate(table))
print(tabulate(table, headers=["k_cluster", "Cluster", "Lambda", "Logistic", "Majority", "Baseline"],tablefmt="latex"))    
