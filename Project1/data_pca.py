# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:36:56 2019

@author: Michael
"""
from main import *

######
# Cummulative
######

# Extract class names to python list,
# then encode with integers (dict)
sexLabels = df.sex
sexNames = sorted(df.sex.unique())
sexDict = dict(zip(sexNames, range(5)))

# Extract vector y, convert to NumPy array
y = np.asarray([sexDict[value] for value in list(sexLabels)])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((4177, 5))
for i, col_id in enumerate(range(3, 11)):
    X[:, i] = np.asarray(doc.col_values(col_id, 2, 92))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

######
#CUMMULATIVE; DID NOT WORK!
######
X1= df[attributeNamesC]
N = 6
# Subtract mean value from data
MEAN = [0.523992,0.407881,0.139516,0.828742,0.238831,9.933684]
STD = [0.120093,0.099240,0.041827,0.490389,0.139203,3.224169]
Y = X - np.ones((len(X),len(attributeNamesC)))*MEAN

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#####
#DOwn to one
######





