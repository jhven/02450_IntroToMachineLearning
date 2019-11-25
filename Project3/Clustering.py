from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, legend
from sklearn import model_selection, preprocessing
from sklearn.mixture import GaussianMixture
import numpy as np
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


#From exercise 6_2_1 
# See algorithm 6, page 173 for insiparation 
#Import data
from main import *

# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

######################################################################
# NOTICE -  THE FOLLOWING CODE IS CURRENTLY MADE FOR THE WHOLE DATA SET
#           I.E. THE INITIALLY CATEGORIZED OUTLIERS ARE IN THE DATA SET!
######################################################################

################################################
# EXCLUSION OF OULIERS
################################################
df = df[df.Height < 0.4]
df = df[df.Height > 0]
df = df[df.Rings < 26]

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

##### Intervals of number of rings
# Grouping all observations in intervals of 5 in the number of rings
Newdf['RingsGroup'] = '01-05'
Newdf.loc[Newdf['Rings'] >  5,'RingsGroup'] = '06-10'
Newdf.loc[Newdf['Rings'] > 10,'RingsGroup'] = '11-15'
Newdf.loc[Newdf['Rings'] > 15,'RingsGroup'] = '16-20'
Newdf.loc[Newdf['Rings'] > 20,'RingsGroup'] = '21-25'
Newdf.loc[Newdf['Rings'] > 25,'RingsGroup'] = '26-30'

##### Standardizing the data #####
# Get column names first
names = Newdf.columns[1:len(Newdf.columns)-1]
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(Newdf.iloc[:, 1:len(Newdf.columns)-1])
scaled_df = pd.DataFrame(scaled_df, columns=names)
Newscaleddf = pd.DataFrame({"Sex":list(Newdf["Sex"]),"Length":list(scaled_df["Length"]),"Diam":list(scaled_df["Diam"]),"Height":list(scaled_df["Height"]),"Whole":list(scaled_df["Whole"]),"Shucked":list(scaled_df["Shucked"]),"Viscera":list(scaled_df["Viscera"]),"Shell":list(scaled_df["Shell"]),"Rings":list(scaled_df["Rings"]),"RingsGroup":list(Newdf["RingsGroup"])}, dtype ="d")

# Extract vector y, convert to NumPy array
y = Newdf.RingsGroup.squeeze().to_numpy()

# Creating matrix X, only for the attributes of interest
X = Newscaleddf[attributeNamesC[:5]]

# Computing M, N and C
N, M = X.shape
C = len(np.unique(Newscaleddf.RingsGroup))



# Perform hierarchical/agglomerative clustering on data matrix
#Method = 'single'
Method = 'complete'
#Method = 'centroid'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = C
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1,figsize=(15,15))
xlabel('SOME GOOD AXIS NAME')
ylabel('SOME GOOD AXIS NAME')
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
xlabel('SOME GOOD AXIS NAME')
ylabel('SOME GOOD AXIS NAME')
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()

# Calculate accuracy
accuracy_hierarchical = sum([cls[i] == y_classNames[i] for i in range(len(cls))]) / N
print("Accuracy of the heirarchical clustering:", accuracy_hierarchical)


################################################
# Gaussian Mixture Model
################################################
#
## Range of K's to try
#KRange = range(max(1, C - 5),C + 10)
#T = len(KRange)
#
#covar_type = 'full'       # you can try out 'diag' as well
#reps = 3                  # number of fits with different initalizations, best result will be kept
#init_procedure = 'kmeans' # 'kmeans' or 'random'
#
## Allocate variables
#BIC = np.zeros((T,))
#AIC = np.zeros((T,))
#CVE = np.zeros((T,))
#
## K-fold crossvalidation
#CV = model_selection.KFold(n_splits=10,shuffle=True)
#
#for t,K in enumerate(KRange):
#        print('Fitting model for K={0}'.format(K))
#
#        # Fit Gaussian mixture model
#        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
#                              n_init=reps, init_params=init_procedure,
#                              tol=1e-6, reg_covar=1e-6).fit(X)
#        
#        # Get BIC and AIC
#        BIC[t,] = gmm.bic(X)
#        AIC[t,] = gmm.aic(X)
#
#        # For each crossvalidation fold
#        for train_index, test_index in CV.split(X):
#
#            # extract training and test set for current CV fold
#            X_train = X.to_numpy()[train_index,:]
#            X_test = X.to_numpy()[test_index,:]
#
#            # Fit Gaussian mixture model to X_train
#            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)
#
#            # compute negative log likelihood of X_test
#            CVE[t] += -gmm.score_samples(X_test).sum()
#            
#
## Plot results
#figure(3);
#plot(KRange, BIC,'-*b')
#plot(KRange, AIC,'-xr')
#plot(KRange, 2*CVE,'-ok')
#legend(['BIC', 'AIC', 'Crossvalidation'])
#xlabel('K')
#show()