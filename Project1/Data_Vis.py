# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:27:50 2019

@author: Michael
"""

from main import * 

##########################
#BOX PLOT 
############################

sns.boxplot(data=df[attributeNamesC[:-1]],color="white")

########################
# Normal distribution 
###########################
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
    
    # Over the histogram, plot the theoretical probability distribution function:
    x = np.linspace(attributeV.min(), attributeV.max(), N)
    pdf = stats.norm.pdf(x,loc=mu,scale=s)
    plot(x,pdf,'.',color='red')
    xlabel(item)
    ylabel("density")
    show()


##################################################
#Correlation 
#####################################################

# Create the default pairplot
sns.pairplot(df[attributeNamesC], x_vars=attributeNamesC,y_vars=attributeNamesC)

#Correlation of the plots 
COR = df[attributeNamesC].corr()
#print(tabulate(COR, tablefmt="latex", floatfmt="2"))