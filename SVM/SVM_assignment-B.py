##############################################################################
#### import libraries
import pandas as pd
import numpy as np
df=pd.read_csv('D:/data science/assignments_csv/forestfires.csv')
df
df.shape

df.info()
df.describe()
df.isna().sum()

df1=df.drop_duplicates()
df1

##############################################################################
################ EDA
import seaborn as sns 
sns.set_style(style='darkgrid')
sns.pairplot(df1)
sns.countplot(x='size_category',data=df1)

###########################################################################
###### Data Transformations 
### Standardscalar on continious variables
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
for column in df1.columns:
    if df1[column].dtype == 'object':
        continue
    df1[column]=SS.fit_transform(df1[[column]])
df1    
### Label Encoder on discrete variables
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in df1.columns:
    if df1[column].dtype == np.number:
        continue
    df1[column]=LE.fit_transform(df1[column])
    
df1    
#############################################################################
#### Splitting the variables
X=df1.iloc[:,0:30]
X
Y=df1['size_category']
Y

#### Data partition
### Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y)


##############################################################################
########## Model fitting
''' For support vector classifier we have three types of kernals which are 
linear,poly,rbf .we will fit 3 svm model based on their kernals'''

#### Support Vector classifier 
### Kernal='linear'
from sklearn.svm import SVC
clf = SVC(kernel='linear',C=5.0)
clf.fit(X_train, Y_train)

### Y predictions
Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)

### accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

##############################################################################
#### Support Vector classifier
### Kernal='Poly
from sklearn.svm import SVC
clf = SVC(kernel='poly',degree=5.0)
clf.fit(X_train, Y_train)

### Y prediictions
Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)

### accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

##############################################################################
#### Support Vector classifier
### Kernal=rbf
from sklearn.svm import SVC
clf = SVC(kernel='rbf',gamma='scale')
clf.fit(X_train, Y_train)

### Y predictions
Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)

#### accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

''' Kernal=linear has best accuracy when compared to kernal=linear and
 kernal=gama,so we finiaize kernal=poly with accuracy= 
                                                Training Accuracy score: 98.16
                                                Test Accuracy score: 96.88'''
                                               









