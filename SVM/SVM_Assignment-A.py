##############################################################################
#### import libraries
import pandas as pd
import numpy as np

#### import datasets
df_train=pd.read_csv('D:/data science/assignments_csv/SalaryData_Train(1).csv')
df_test=pd.read_csv('D:/data science/assignments_csv/SalaryData_Test(1).csv')

####  shape of the datasets 
df_train.shape
df_test.shape
############################################################################
#### Train Dataset
df_train.info()         
df_train.describe()     
df_train.isna().sum()
df_train['Salary'].value_counts()

#### dropping duplicates
train=df_train.drop_duplicates()
train.head()

##############################################################################
################ EDA
import seaborn as sns 
sns.set_style(style='darkgrid')
sns.pairplot(train)

##### Countplot for discreate Variables
sns.countplot(x='Salary',data=train)
sns.countplot(x='workclass',data=train)
sns.countplot(x='education',data=train)
sns.countplot(x='maritalstatus',data=train)
sns.countplot(x='occupation',data=train)
sns.countplot(x='relationship',data=train)
sns.countplot(x='race',data=train)
sns.countplot(x='sex',data=train)
sns.countplot(x='native',data=train)


##############################################################################
############### Data transformations on Train dataset
#### Labelencoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column in train.columns:
    if train[column].dtype == np.number:
        continue
    train[column]=LE.fit_transform(train[column])

train.info()

###############################################################################
#### Test Dataset
df_test.info()
df_test.describe()
df_test.isna().sum()
df_test['Salary'].value_counts()


#### dropping duplicates 
test=df_test.drop_duplicates()
test.head()

################### Test-data EDA
import seaborn as sns 
sns.set_style(style='darkgrid') 
sns.pairplot(test)

##### Countplot for Target variable
sns.countplot(x='Salary',data=test)
sns.countplot(x='workclass',data=train)
sns.countplot(x='education',data=train)
sns.countplot(x='maritalstatus',data=train)
sns.countplot(x='occupation',data=train)
sns.countplot(x='relationship',data=train)
sns.countplot(x='race',data=train)
sns.countplot(x='sex',data=train)
sns.countplot(x='native',data=train)


##############################################################################
############### Data transformations on Test dataset
#### Labelencoder
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in test.columns:
    if test[column].dtype == np.number:
        continue
    test[column]=LE.fit_transform(test[column])

test.info()

##############################################################################
#### X and Y Train variables of Train dataset
X_train=train.iloc[:,0:13]
X_train
Y_train=train['Salary']
Y_train

#### X and Y Test variables of Test dataset
X_test=test.iloc[:,0:13]
X_test
Y_test=test['Salary']
Y_test

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

''' Kernal=poly has best accuracy when compared to kernal=linear and
 kernal=gama,so we finiaize kernal=poly with accuracy= 
                                               Training Accuracy score: 82.37
                                               Test Accuracy score: 81.99'''

