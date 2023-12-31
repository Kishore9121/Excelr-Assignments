###############################################################################
##### Import Libraries
import pandas as pd
import numpy as np

#### Import dataset
df=pd.read_csv('D:/data science/assignments_csv/wine.csv')
df
df.head()
###############################################################################
#### Droping the type column
X=df.drop('Type',axis=1)
X

'''As mentioned in the problem statement to drop the class column'''

###############################################################################
#### Data Transformation
### StandardScalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for column in X.columns:
    if X[column].dtype=='object':
        continue
    X[column]=scaler.fit_transform(X[[column]])
X

###############################################################################
#### Principal Compound Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
PC = pca.fit_transform(X)

### Creating a data frame to observe the variance
df1 = pd.DataFrame(pca.explained_variance_ratio_)
df1

### Taking first 3 principal components and creaating a dataframe
pca_df = pd.DataFrame(data=PC, columns=['PC1', 'PC2', 'PC3'])
pca_df

###############################################################################
###### AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
Y = cluster.fit_predict(pca_df)
Y

Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts

###############################################################################
#### Initializing KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,n_init=20)
kmeans = kmeans.fit(pca_df) # Fitting with inputs
# Predicting the clusters
Y = kmeans.predict(pca_df)
Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts

#### Total with in centroid sum of squares 
kmeans.inertia_
clust = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(pca_df)
    clust.append(kmeans.inertia_)

##### Elbow method 
import maatplotlib.pyplot as plt
plt.scatter(x=range(1,11), y=clust,color='red')
plt.plot(range(1,11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

##### Comparing between PCA-based clusters with original "Type" column
#### Using AgglomerativeClustering
df['PCA_Cluster'] = Y_new 

#### Using groupby function for Type and PCA_Cluster for comparing average 
cluster_comparison = df.groupby(['Type', 'PCA_Cluster']).mean()
cluster_comparison
