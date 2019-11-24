
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
# REGRESSION
##########################






##########################
# CLASSIFICATION
##########################

