# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:04:43 2019

@author: Michael
"""


from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats




#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

#Transform the data into prober format 

X = df_noOutliers[["Length","Diam","Height","Whole","Shell"]].get_values()
#Normalize? 
X = stats.zscore(X);
y = df_noOutliers['Rings'].get_values()
y.shape=[len(y),1]
attributeNames = ["Length","Diam","Height","Whole","Shell"]
N, M = X.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10 #Outer cross validation 
K2 = 10 #Inner cross validation 

K = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K2,shuffle=True)


Error_Baseline = list()
Error_Linear = list()
Linear_Lambda = list()
Error_ANN = list()
Unit_ANN = list()

Regulization = np.logspace(-5, 5, 10)
#Regulization = [0.1,.2,.3,.4,.5,.6,.7,.8,.9,.1]

#####################################################
#ann stuff
#####################################################

# Parameters for neural network classifier
n_hidden_units = [20,15,10,5,1]    # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 5000        # 

# Define the model
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

errors = [] # make a list for storing generalizaition error in each loop


k1 = 0 
#Outer loop 
for train_index, test_index in CV1.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    X_train_ANN = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train_ANN = torch.tensor(y[train_index], dtype=torch.float)
    X_test_ANN = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test_ANN = torch.tensor(y[test_index], dtype=torch.uint8)
    
    ###########################
    #    Initialization       #
    ###########################    
    Error_Baseline_Inner = list()
    Error_Baseline_mean = list()
    
    Error_Linear_Inner = list()
    Error_Linear_Inner_reg = list()

    
    Error_Ann_Inner = list()
    Error_Ann_Hidden_units = list()

    # Inner Loop 
    for train_index2, test_index2 in CV2.split(X_train):
        
        #Extract new training set, of the current one. 
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2] 
        
        ###########################
        #     LINEAR REGRESSION   #
        ###########################
        
        #Test different regulaization factors
        Error_m_Inner = list()
        for item in Regulization:
            m = lm.Ridge(fit_intercept=True,alpha=item).fit(X_train2, y_train2)
            
            #Take the best of the inner ()
            Error_m_Inner.append( ((y_test2-m.predict(X_test2))**2).sum()/y_test2.shape[0])
        
        Error_Linear_Inner.append(min(Error_m_Inner))
        Error_Linear_Inner_reg.append(Regulization[np.argmin(Error_m_Inner)])

           
        
        ###########################
        #          ANN            #
        ###########################
        
        
        X_train_ANN2 = torch.tensor(X[train_index2,:], dtype=torch.float)
        y_train_ANN2 = torch.tensor(y[train_index2], dtype=torch.float)
        X_test_ANN2 = torch.tensor(X[test_index2,:], dtype=torch.float)
        y_test_ANN2 = torch.tensor(y[test_index2], dtype=torch.uint8)
        
        Error_h_Inner = list()
        for item in n_hidden_units:
      
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, item), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(item, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
        # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_ANN2,
                                                           y=y_train_ANN2,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
            #Determine estimated class labels for test set
            y_test_est2 = net(X_test_ANN2)
            
            se = (y_test_est2.float()-y_test_ANN2.float())**2
            mse = (sum(se).type(torch.float)/len(y_test_ANN2)).data.numpy()
            Error_h_Inner.append(mse)
  

        Error_Ann_Inner.append(min(Error_h_Inner))
        Error_Ann_Hidden_units.append(n_hidden_units[np.argmin(Error_h_Inner)])    
    
            
        
        ###########################
        #     BASELINE            #
        ###########################
        
        Baseline_y_train = y_test2.mean()

        Error_Baseline_Inner.append(((y_test2-Baseline_y_train)**2).sum()/y_test2.shape[0])
        Error_Baseline_mean.append(Baseline_y_train)
    ######################################################################
    # Test model on full set 
    ######################################################################

    ###########################
    #     LINEAR REGRESSION   #
    ###########################        
    #Pick best model 

    best_reg = Error_Linear_Inner_reg[np.argmin(Error_Linear_Inner)]
    m = lm.Ridge(fit_intercept=True,alpha=best_reg).fit(X_train, y_train)
    
    Error_Linear.append( ((y_test-m.predict(X_test))**2).sum()/y_test.shape[0])
    Linear_Lambda.append(best_reg)
    
    
    ###########################
    #          ANN            #
    ###########################
    best_unit = Error_Ann_Hidden_units[np.argmin(Error_Ann_Inner)]
    
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, best_unit), #M features to n_hidden_units
            torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.Linear(best_unit, 1), # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
            )    
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_ANN,
                                                       y=y_train_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test_ANN)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test_ANN.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
    Error_ANN.append(float(mse)) # store error rate for current CV fold 
    Unit_ANN.append(best_unit)
    ###########################
    #     BASELINE            #
    ###########################    

    Basline_mean = Error_Baseline_mean[np.argmin(Error_Baseline_Inner)]

    # I AM NOT SURE; IF I JUST SHOULD TAKE THE DIFFERENCE; OR SQUARED ERROR HERE? 
    Error_Baseline.append(((y_test-Basline_mean)**2).sum()/y_test.shape[0])


    print('Cross validation fold {0}/{1}'.format(k1+1,K))
    k1 +=1



#Output to Latex
table =[Unit_ANN, Error_ANN,  Linear_Lambda , Error_Linear ,Error_Baseline ]
table = np.array(table).T.tolist()
print(tabulate(table))

print(tabulate(table, headers=["H","Error_ANN", "Lambda","E_test", "Etest_baseline"],tablefmt="latex"))    


#############################
#############################
#############################
#############################
# Exercise 3, about the setup i  


# perform statistical comparison of the models
def Diff(li1, li2): 
    Z = list()
    for i in range(len(li1)):
        Z.append(abs(float(li1[i])-float(li2[i])))
    
    
    return Z
ANN = [5.02711,5.065858,4.839445,4.835392,5.0271187,5.065858,4.8585243,4.8665185,4.7865295,4.8641644]
Linear = [5.44365,5.83959,4.8351,4.8268,5.65217,5.85463,6.55834,6.44158,5.04122,5.00129]
Basline=[11.3332,11.1856,9.21506,9.10367,9.86837,11.196,11.3596,11.4402,9.52578,10.2317]


# compute confidence interval of model A
alpha = 0.05
z = Diff(Basline,Linear)
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

#Which mean that there are significant difference. 

