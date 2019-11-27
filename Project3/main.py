
#Some code, is taken from the exercises in machinelearning 


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import cov
from scipy import stats
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.linalg import svd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from matplotlib.pyplot import (figure, subplot, plot, legend, show, hist, colorbar,
                               xlabel, ylabel, xticks, yticks, boxplot, setp,
                               title, ylim, subplot, imshow)

# Open file 
attributeNames =["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
# C for correct,
attributeNamesC = ["Length","Diam","Height","Whole","Shell","Rings"]

df = pd.read_csv('../abalone.data',names=attributeNames)
raw_data = df.get_values()

# Add binary attribute of whether abalone is adult (i.e. female or male) or infant
df['Adult'] = df.Sex != 'I'
df.Adult = df.Adult.astype(int)

# Get class names
classLabels = df.Rings
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y_classNames = np.array([classDict[cl] for cl in classLabels])

# Excluding outliers from the data set
df_noOutliers = df[df.Height < 0.4]
df_noOutliers = df_noOutliers[df_noOutliers.Height > 0]


##########################
# FUNCTION DEFINITIONS
##########################

# Scatterplot of data
def scatterplot(X, centroids='None', y='None', covars='None'):
    X = np.asarray(X)
    if type(y) is str and y=='None':
        y = np.zeros((X.shape[0],1))
    else:
        y = np.asarray(y)
    if type(centroids) is not str:
        centroids = np.asarray(centroids)
    C = np.size(np.unique(y))
    ncolors = np.max([C])
    
    # plot data points color-coded by class, cluster markers and centroids
    #hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = plt.cm.jet(color/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y)):
        plt.plot(X[(y==cs).ravel(),0], X[(y==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    if type(centroids) is not str:        
        for cd in range(centroids.shape[0]):
            plt.plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    
    # create legend        
    legend_items = np.unique(y).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
    plt.legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})





##########################
# CLASSIFICATION
##########################

