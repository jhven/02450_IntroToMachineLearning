


# In this exercise we will rely on pandas for some of the processing steps:
import pandas as pd
import numpy as np 
import seaborn as sns
from scipy import stats
from scipy.io import loadmat
from scipy.stats import zscore
import scipy
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
from matplotlib.pyplot import (figure, subplot, boxplot, title, xticks, ylim, 
                               show)
from matplotlib.pyplot import (figure, subplot, plot, legend, show, 
                               xlabel, ylabel, xticks, yticks, boxplot, setp,title,ylim)

#Open file 
attributeNames =["Sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
df = pd.read_csv('abalone.data',names=attributeNames)
raw_data = df.get_values() 

attributeNamesC = ["Length","Diam","Height","Whole","Shell","Rings"]
###############################################################################################
#Basic statistics
#Do not take the first, becasue, it is se
basic_stat = [["name","min", "median", "max", "mean", "std"]]
for item in attributeNames[1:]:
    a = [item,df[item].min() , np.median(df[item]) ,df[item].max(), 
         df[item].mean(), df[item].std(ddof=1)]

    #basic_stat.concat(a)
    basic_stat.append(a)
   
#print(tabulate(basic_stat, tablefmt="latex", floatfmt="2"))    
df[attributeNamesC].describe()
print(tabulate(df[attributeNamesC].describe(), tablefmt="latex", floatfmt="2"))       

  







    