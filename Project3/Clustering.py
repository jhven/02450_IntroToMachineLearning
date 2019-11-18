from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, preprocessing
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import scipy.stats as st
import operator
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Transform the data into prober format 

# Add binary attribute of whether abalone is adult (i.e. female or male) or infant
#df_noOutliers['Adult'] = df_noOutliers.Sex != 'I'
#df_noOutliers.Adult = df_noOutliers.Adult.astype(int)

# Extract class names to python list,
# then encode with integers (dict)

######################################################################
# NOTICE -  THE FOLLOWING CODE IS CURRENTLY MADE FOR THE WHOLE DATA SET
#           I.E. THE INITIALLY CATEGORIZED OUTLIERS ARE IN THE DATA SET!
######################################################################

################################################
# EXCLUSION OF OULIERS
################################################
#df_noOutliers = df[df.Height < 0.4]
#df_noOutliers = df_noOutliers[df_noOutliers.Height > 0]
# to use this instead make sure to use it instead of 'df' in all places below!

################################################
# FEATURE EXTRACTION 
################################################

##### One-out-of-K coding ######
Infant = list()
Female = list()
Male = list()

for i in list(df["Sex"]):
    if i == "I":
        Infant.append(1)
    else:
        Infant.append(0)
    if i == "F":
        Female.append(1)
    else:
        Female.append(0)
    if i == "M":
        Male.append(1)
    else:
        Male.append(0)

Newdf = pd.DataFrame({"Sex": list(df["Sex"]),"Length":list(df["Length"]),"Diam":list(df["Diam"]),"Height":list(df["Height"]),"Whole":list(df["Whole"]),"Shucked":list(df["Shucked"]),"Viscera":list(df["Viscera"]),"Shell":list(df["Shell"]),"Rings":list(df["Rings"])}, dtype ="d")
attributeNames = list(Newdf.columns)

##### Standardizing the data #####
# Get column names first
names = Newdf.columns[1:]
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(Newdf.iloc[:, 1:])
scaled_df = pd.DataFrame(scaled_df, columns=names)
Newscaleddf = pd.DataFrame({"Sex":list(df["Sex"]),"Length":list(scaled_df["Length"]),"Diam":list(scaled_df["Diam"]),"Height":list(scaled_df["Height"]),"Whole":list(scaled_df["Whole"]),"Shucked":list(scaled_df["Shucked"]),"Viscera":list(scaled_df["Viscera"]),"Shell":list(scaled_df["Shell"]),"Rings":list(scaled_df["Rings"])}, dtype ="d")

# Extract vector y, convert to NumPy array
y = Newscaleddf.Sex.squeeze().to_numpy()

# Creating matrix X, only for the attributes of interest
X = Newscaleddf[attributeNamesC]

# Computing M, N and C
N, M = X.shape
C = len(classNames)



# Perform hierarchical/agglomerative clustering on data matrix
#Method = 'single'
Method = 'complete'
#Method = 'centroid'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 3
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
xlabel('SOME GOOD AXIS')
ylabel('SOME GOOD AXIS')
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
xlabel('SOME GOOD AXIS')
ylabel('SOME GOOD AXIS')
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
