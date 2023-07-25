###############################################################################
##### Import libraries
import pandas as pd 
import numpy as np

#### Import dataset
df=pd.read_csv('D:/data science/assignments_csv/crime_data.csv')
df

#############################################################################
df.shape ### shape of the data
df.info() ### info of data
df.describe() ### describing data

##############################################################################
#### EDA
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

##############################################################################
#### reaname the data
df.rename(columns={'Unnamed: 0':'city'},inplace=True)
df

############################################################################
##### Data Transformations
### Labelencoder on nominal data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['city']=LE.fit_transform(df['city'])
df

### Standardscalar on continious data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df['Murder']=ss.fit_transform(df[['Murder']])
df['Assault']=ss.fit_transform(df[['Assault']])
df['UrbanPop']=ss.fit_transform(df[['UrbanPop']])
df['Rape']=ss.fit_transform(df[['Rape']])
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
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y) ### creating Y dataframe
Y_new.value_counts() ### Y value counts

###############################################################################
##### Method = complete
#### Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))

###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts
###############################################################################
##### Method = averagee
#### Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average'))

###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
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
kmeans = KMeans(n_clusters=4,n_init=20)

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

''' For Kmeans clustering clusters as 4 and 5 is giving best elbow curve .so 
    i have taken no of clusters as 4'''

##############################################################################
##### DBSCAN
X = df.iloc[:,1:].values 
X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.25, min_samples=3)
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
Finaldata = df[(df["Cluster id"] == 0)| (df["Cluster id"] == 1)].reset_index(drop=True)
Finaldata

''' For DBSCAN i have taken eps=1.25 because below 1.25 more noise points are 
    occuring after that i kept the noise points out and prepared a new final
    data which has other cluster ids 0's and 1's and kept them in a correct
    indexing '''



























