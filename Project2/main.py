
#Some code, is taken from the exercises in machinelearning 


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.linalg import svd
from tabulate import tabulate
from matplotlib.pyplot import (figure, subplot, plot, legend, show, hist,
                               xlabel, ylabel, xticks, yticks, boxplot, setp,title,ylim,subplot)

# Open file 
attributeNames =["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
# C for correct,
attributeNamesC = ["Length","Diam","Height","Whole","Shell","Rings"]

df = pd.read_csv('../abalone.data',names=attributeNames)
raw_data = df.get_values()

# Excluding outliers from the data set
df_noOutliers = df[df.Height < 0.4]
df_noOutliers = df_noOutliers[df_noOutliers.Height > 0]


##########################
# REGRESSION
##########################






##########################
# CLASSIFICATION
##########################

# Add binary attribute of whether abalone is adult (i.e. female or male) or infant
df_noOutliers['Adult'] = df_noOutliers.Sex != 'I'
df_noOutliers.Adult = df_noOutliers.Adult.astype(int)

# Extract vector y, convert to NumPy array
y = df_noOutliers.Adult.squeeze()

# Creating matrix X, only for the attributes of interest
X = df_noOutliers[attributeNamesC]

# Computing M, N and C
C = len(df_noOutliers.Adult.unique())

# Define training set and test set from the dataframe
max_training_index = 3000

X_train = X.iloc[:max_training_index].to_numpy()
X_test = X.iloc[max_training_index:].to_numpy()
y_train = y.iloc[:max_training_index].to_numpy()
y_test = y.iloc[max_training_index:].to_numpy()

# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])

# K-nearest neighbors
K = 5





K=3

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
#metric = 'cosine' 
#metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
#metric='mahalanobis'
#metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)


# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
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
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()