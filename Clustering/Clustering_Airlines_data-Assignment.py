###############################################################################
##### Import libraries
import pandas as pd 
import numpy as np

#### Import dataset
df=pd.read_excel('D:/data science/assignments_csv/EastWestAirlines.xlsx',sheet_name='data')
df
df.head()
df.columns

#############################################################################
df.shape ### shape of the data
df.info() ### info of data
df.describe() ### describing data
df.columns

##############################################################################
#### EDA
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

##############################################################################
#### reaname the data
df.rename(columns={'ID#':'ID','Award?':'Award'},inplace=True)
df.columns

############################################################################
##### Data Transformations
### Standardscalar on continious data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
for column in df.columns:
    if df[column].dtype=='object':
        continue
    df[column]=ss.fit_transform(df[[column]])
    
df
###############################################################################
### creating X variables
X=df.iloc[:,1:]
X

###############################################################################
##### Merthod=single
#### Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='single'))

###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y) ### creating Y dataframe
Y_new.value_counts() ### Y value counts

###############################################################################
##### Method = complete
#### Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))

###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts
###############################################################################
##### Method = averagee
#### Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average'))

###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts

''' For AgglomerativeClustering i tried with all the three different methods 
    such as single , complete and average. Amoung these methods complete linkage 
    is best clusers '''


##############################################################################
#### Initializing KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,n_init=20)

kmeans = kmeans.fit(X) # Fitting with inputs
# Predicting the clusters
Y = kmeans.predict(X)
Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts

#### Total with in centroid sum of squares 
kmeans.inertia_

clust = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)

##### Elbow method 
plt.scatter(x=range(1,11), y=clust,color='red')
plt.plot(range(1,11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

##############################################################################
##### DBSCAN
X = df.iloc[:,1:].values 
X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan.fit(X) ### fitting dbscan

Y = dbscan.labels_
pd.DataFrame(Y).value_counts()

### creating cluster id with dataframe
df["Cluster id"] = pd.DataFrame(Y)
df.head()

#### Checking the noise points
noise_points = df[df["Cluster id"] == -1]
noise_points

#### final data
Finaldata = df[(df["Cluster id"] == 0)| (df["Cluster id"] == 1)
               |(df["Cluster id"] == 2)].reset_index(drop=True)
Finaldata

''' For DBSCAN i have taken eps=3 because below 3 more noise points are 
    occuring which i do not want that much of outliers after that i kept
    the noise points out and prepared a new final data which has other
    cluster ids 0's,1's and 2's and kept them in a correct indexing '''






