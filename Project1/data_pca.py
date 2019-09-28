# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:36:56 2019

@author: Michael
"""
print("new")
from main import *

######
#CUMMULATIVE; DID NOT WORK!
######
X = df[attributeNamesC]
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





