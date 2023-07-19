##########################################################################
###### Import Libraries
import pandas as pd
import numpy as np

###### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/Company_Data.csv')
df
df.columns
df['Sales'].max()
df['Sales'].min()
df['Sales'].median()
df['Sales'].mean()

'''df["sales"]="small"
df.loc[df["Sales"]>7.49,"sales"]="large"
df.drop(["Sales"],axis=1,inplace=True)
df'''


for index, row in df.iterrows():
    taxable_income = row['Sales']
    if taxable_income <= 7.49:
        classification = "Low"
    else:
        classification = "High"
    df.at[index, 'Target'] = classification
df
df.info()
##########################################################################
###### EDA
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

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
X=df.iloc[:,0:11]
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

