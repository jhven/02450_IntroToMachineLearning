
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

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
i = 0
u = 0 
for train_index, test_index in CV1.split(X):
    i += 1
    print(i)
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    

    #Inner validation loop 
    
    Error_train2 = np.empty((K,1))
    Error_test2 = np.empty((K,1))
    Error_Smallest = None 
    k2 = 0 
    for train_index2, test_index2 in CV2.split(X_train):
        ###########################
        #     LINEAR REGRESSION   #
        ###########################
        #Extract new training set, of the current one. 
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]     
        
        
        m = lm.LinearRegression(fit_intercept=True).fit(X_train2, y_train2)
        Error_train2[k2] = np.square(y_train2-m.predict(X_train2)).sum()/y_train2.shape[0]
        Error_test2[k2] = np.square(y_test2-m.predict(X_test2)).sum()/y_test2.shape[0]

       
        
        k2 += 1 
        
        #Possible to input different models here, (which we do not do )
        for S in range(1): 
            pass 
    


    
        
    # Pick which train data, to use for test.  
        
    #REMARK; THE POINT WITH TRAIN MODEL AGAIN IS TAKEN OUT; BECAUSE ONLY ONE MODEL IS USED!
        
        # Test if the model is better than the prevoius with validation error
        
        
        
    
    
    # Compute squared error without using the input data at all
    #Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    #Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    #THIS IS HERE THE MAGIN HAPPEND 
    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # I am not sure about the Train mu* on D*par, because should we use the model 
    # we just saw was the best+ 

  
    print('Cross validation fold {0}/{1}'.format(k+1,K))
   # print('Train indices: {0}'.format(train_index))
   # print('Test indices: {0}'.format(test_index))
   # print('Features no: {0}\n'.format(selected_features.size))

    k+=1


# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
#print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
#print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')



    
show()