##########################################################################
###### Import Libraries
import pandas as pd
import numpy as np

###### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/Fraud_check.csv')
df

###########################################################################
df.shape ### Shape of the dataset
df.columns ### columns of the dataset
df.info() ### info of the Dataset

##########################################################################
''' Here the target variable is given  indirectly which is taxable income, 
    so taxable_income<=30000 considering as risk and taxable_income>30000 
    considering as good.for that i am itterarting through the rows of 
    taxable_income and creating target variable using For Loop'''

for index, row in df.iterrows():
    taxable_income = row['Taxable.Income']
    if taxable_income <= 30000:
        classification = "risk"
    else:
        classification = "good"
    df.at[index, 'Target'] = classification
df
###########################################################################
df['Target'].value_counts() ### Value counts of Target
df['Undergrad'].value_counts() ### Value counts of Undergrad
df['Marital.Status'].value_counts() ### Value counts of Marital status
df['Urban'].value_counts() ### Value counts of Urban

##########################################################################
###### EDA
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

#### Countplot of all discrete variables
sns.countplot(x='Target',data=df)
sns.countplot(x='Undergrad',data=df)
sns.countplot(x='Marital.Status',data=df)
sns.countplot(x='Urban',data=df)

###########################################################################
###### Data Transformations
### Standard Scalar on Continious Variables
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
for column in df.columns:
    if df[column].dtype == object:
        continue
    df[column]=SS.fit_transform(df[[column]])
df

### Labelencoding on discrete variables
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column]=LE.fit_transform(df[column])
df
df.info()

###########################################################################
### Splitting Varibles
X=df.iloc[:,0:6]
X
Y=df['Target']
Y
##### Data partition
#### Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

###########################################################################
##### model fitting
### DecisionTreeClassifier
### criterion='entropy'
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,Y_train)

###########################################################################
#### model predictions
Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)

###########################################################################
### metrics
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

############################################################################
##### criterion='entropy'
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)

###########################################################################
#### model predictions
Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)

###########################################################################
### metrics
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

''' I have fitted the model with both gini and entropy criterion
    in both cases i got accuracy score as 100'''
    
''' So the accuracy score is 100,hence we need not to perform 
   ensemnle methods like bagging,randomforest,ada boost etc....'''






