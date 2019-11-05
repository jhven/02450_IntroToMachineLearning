# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:18:32 2019

@author: Michael
"""
from main import * 

# Extract vector y, convert to NumPy array
y = df_noOutliers.Adult.squeeze().to_numpy()

# Creating matrix X, only for the attributes of interest
X = df_noOutliers[attributeNamesC].to_numpy()

# Computing M, N and C
N, M = X.shape
C = len(df_noOutliers.Adult.unique())

# Define training set and test set from the dataframe
max_training_index = 3000

X_train = X[:max_training_index]
X_test = X[max_training_index:]
y_train = y[:max_training_index]
y_test = y[max_training_index:]

# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])
title('Distribution of \'Adult\' class');

show()


##########################
### K-nearest neighbors
print("\n\n --- Classification with method: K-nearest neighbours\n")
K = 12

# Distance variables
dist=2
metric = 'minkowski'
metric_params = {}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)

# Plot the classfication results
styles = ['ob', 'or']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0:.4f}%, Error Rate: {1:.4f}%)'.format(accuracy, error_rate));

show()

#### THE FOLLOWING IS USED TO DETERMINE THE BEST NUMBER OF NEIGHBOURS FOR KNN ! ! !
## Maximum number of neighbors
#L=40
#
#CV = model_selection.LeaveOneOut()
#errors = np.zeros((N,L))
#i=0
#for train_index, test_index in CV.split(X, y):
#    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
#    
#    # extract training and test set for current CV fold
#    X_train = X[train_index,:]
#    y_train = y[train_index]
#    X_test = X[test_index,:]
#    y_test = y[test_index]
#
#    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
#    for l in range(1,L+1):
#        knclassifier = KNeighborsClassifier(n_neighbors=l);
#        knclassifier.fit(X_train, y_train);
#        y_est = knclassifier.predict(X_test);
#        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
#
#    i+=1
#    
## Plot the classification error rate
#figure()
#plot(100*sum(errors,0)/N)
#xlabel('Number of neighbors')
#ylabel('Classification error rate (%)')
#show()
### CODE FOR DETERMINATION OF NEIGHBOURS IN KNN ENDS HERE ! ! !


##########################
### Baseline
print("\n\n --- Classification with method: Baseline\n")

unique_baseline, counts_baseline = np.unique(y_train, return_counts=True)
baseline_count_dict = dict(zip(unique_baseline, counts_baseline))
print("Training data has", baseline_count_dict[0], "observations of 'Adult' = 0.")
print("Training data has", baseline_count_dict[1], "observations of 'Adult' = 1.")

# Calculate accuracy of the baseline method
accuracy_baseline = sum(y_test == 1)/len(y_test)
print("The accuracy of the baseline method is:", '{0:.3f}'.format(accuracy_baseline))
print("Thus the error rate of the baseline method is:", '{0:.3f}'.format(1 - accuracy_baseline))