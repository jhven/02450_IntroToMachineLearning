


# In this exercise we will rely on pandas for some of the processing steps:
import pandas as pd
import numpy as np 
#Open file 
attributeNames =["sex","Length","Diam","Height","Whole","Shucked","Viscera","Shell","Rings"]
df = pd.read_csv('abalone.data',names=attributeNames)
raw_data = df.get_values() 


###############################################################################################
#Basic statistics
#Do not take the first, becasue, it is se
basic_stat = [["name","min", "median", "max", "mean", "std"]]
for item in attributeNames[1:]:
    a = [item,df[item].min() , np.median(df[item]) ,df[item].max(), 
         df[item].mean(), df[item].std(ddof=1)]
 
    #basic_stat.concat(a)
    basic_stat.append(a)
   
print(basic_stat)    
###CORRELATION 
    
import seaborn as sns
# Create the default pairplot
#TAGER LIDT LANG TID; saa slaaet fra
#sns.pairplot(df)
  
    