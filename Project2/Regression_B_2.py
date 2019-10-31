# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:21:00 2019

@author: Michael
"""



#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

#Transform the data into prober format 
X = df[["Length","Diam","Height","Whole","Shell"]].get_values()
y = df['Rings'].get_values()

attributeNames = ["Length","Diam","Height","Whole","Shell"]
N, M = X.shape


## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10 #Outer cross validation 
K2 = 10 #Inner cross validation 

K = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True)
CV2 = model_selection.KFold(n_splits=K1,shuffle=True)


Error_Baseline = list()
Error_Linear = list()
Linear_Lambda = list()
Regulization = np.logspace(-6, 6, 10)
Regulization = [0.1,.2,.3,.4,.5,.6,.7,.8,.9,.1]


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
    Error_Baseline_Inner = list()
    Error_Baseline_mean = list()
    
    Error_Linear_Inner = list()
    Error_Linear_Inner_reg = list()
    
        
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
            m = linear_model.Ridge(fit_intercept=True,alpha=item).fit(X_train2, y_train2)
            
            #Take the best of the inner ()
            Error_m_Inner.append( np.square(y_test2-m.predict(X_test2)).sum()/y_test2.shape[0])
        
        Error_Linear_Inner.append(min(Error_m_Inner))
        Error_Linear_Inner_reg.append(Regulization[np.argmin(Error_m_Inner)])

           
        
        ###########################
        #          ANN            #
        ###########################



        ###########################
        #     BASELINE            #
        ###########################
        
        Baseline_y_train = y_test2.mean()

        Error_Baseline_Inner.append(abs(np.square(abs(y_test2-Baseline_y_train)).sum()/y_test2.shape[0]))
        Error_Baseline_mean.append(Baseline_y_train)
    ######################################################################
    # Test model on full set 
    ######################################################################

    ###########################
    #     LINEAR REGRESSION   #
    ###########################        
    #Pick best model 
    best_reg = np.argmin(Error_Linear_Inner_reg)       
    m = linear_model.Ridge(fit_intercept=True,alpha=best_reg).fit(X_train, y_train)
    
    Error_Linear.append( np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0])
    Linear_Lambda.append(Regulization[best_reg])
    
    
    ###########################
    #          ANN            #
    ###########################



    ###########################
    #     BASELINE            #
    ###########################    

    Basline_mean = Error_Baseline_mean[np.argmin(Error_Baseline_Inner)]

    # I AM NOT SURE; IF I JUST SHOULD TAKE THE DIFFERENCE; OR SQUARED ERROR HERE? 
    Error_Baseline.append(abs(np.square(abs(y_test-Basline_mean)).sum()/y_test.shape[0]))


    print('Cross validation fold {0}/{1}'.format(k1+1,K))
    k1 +=1



#Output to Latex
table =[  Linear_Lambda , Error_Linear ,Error_Baseline ]
table = np.array(table).T.tolist()
#print(tabulate(table))

print(tabulate(table, headers=["Lambda","E_test", "Etest_baseline"],tablefmt="latex"))    
   