
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

#Open file 
attributeNames =["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
#C for correct,
attributeNamesC = ["Length","Diam","Height","Whole","Shell","Rings"]

df = pd.read_csv('../abalone.data',names=attributeNames)
raw_data = df.get_values() 

###############################################################################################
#Basic statistics

basic_stat = df[attributeNamesC].describe()
print(tabulate(basic_stat, tablefmt="latex", floatfmt="2"))       

########################## 
##########################
#BOX PLOT 
##########################

sns.boxplot(data=df[attributeNamesC[:-1]],color="white")

########################
# Normal distribution 
########################
for item in attributeNamesC:
    attributeV = df[item]
    
    # Samples, mean, sd 
    N = len(attributeV)
    mu = attributeV.mean()
    s = attributeV.std()
    
    # Number of bins in histogram
    nbins = 20
    
    # Plot the samples and histogram
    figure()
    title('Distribution for '+item)
    hist(attributeV, bins=nbins, density=True)
    
    # Plot the historogram and normal dis
    x = np.linspace(attributeV.min(), attributeV.max(), N)
    pdf = stats.norm.pdf(x,loc=mu,scale=s)
    plot(x,pdf,'.',color='red')
    xlabel(item)
    ylabel("density")
    show()


######################
#Correlation 
######################

# Create the default pairplot
sns.pairplot(df[attributeNamesC], x_vars=attributeNamesC,y_vars=attributeNamesC)

#Correlation of the plots 
COR = df[attributeNamesC].corr()
#print(tabulate(COR, tablefmt="latex", floatfmt="2"))
 
#########################################################################
#########################################################################
#PCA analysis


# Excluding outliers from the data set
df_noOutliers = df[df.Height < 0.4]
df_noOutliers = df_noOutliers[df_noOutliers.Height > 0]

# Extract class names to python list,
# then encode with integers (dict)
sexLabels = df_noOutliers.Sex
sexNames = sorted(df_noOutliers.Sex.unique())
sexDict = dict(zip(sexNames, range(5)))

# Extract vector y, convert to NumPy array
y = np.asarray([sexDict[value] for value in list(sexLabels)])

# Creating matrix X, only for the attributes of interest
X_noOutliers = df_noOutliers[attributeNamesC]

# Computing M, N and C
M = len(attributeNamesC)
N = len(X_noOutliers)
C = len(sexNames)

# Subtract mean value from data
mu = np.asarray(list(X_noOutliers.mean(0)))

Y = X_noOutliers - np.ones((N,M))*mu

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
#plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#####################
### Principal directions of the considered PCA components
#####################

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = Y*(1/np.std(Y,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot of variance explained
plt.figure(figsize=(10,5))
plt.subplots_adjust(hspace=.4)
nrows=1
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum()
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  1+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()

# Make the plot of principal component space
plt.figure(figsize=(10,5))
plt.subplots_adjust(hspace=.4)
nrows=1
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U;
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  1+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNamesC[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')

plt.show()

# Make the plot of projection onto principal component
plt.figure(figsize=(10,5))
plt.subplots_adjust(hspace=.4)
nrows=1
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(sexNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(sexNames)
    plt.axis('equal')

plt.show()
        
         









    