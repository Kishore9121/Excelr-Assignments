###----->>>> import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###------>>>>import dataset
df=pd.read_csv('D:/data science/assignments_csv/Salary_Data.csv')
df

###------->>>> rename dataset
df=df.rename({'YearsExperience':'years_experience','Salary':'salary_hike'},axis=1)
df

###------>>>>>>> EDA
###------>>>>>>> performing scatterplot analysis
plt.scatter(x=df['years_experience'],y=df['salary_hike'],color='red')
plt.xlabel('years of experience') 
plt.ylabel('salary_hike')
plt.show()

###-------->>>> constructing boxplot for verifying outliers
df.boxplot(column='years_experience')
df.boxplot(column='salary_hike')

###-------->>>> analysing correlation
df.corr()

###------->>>> model fitting
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(df[['years_experience']],df['salary_hike'])
LR.intercept_
LR.coef_

###------>>>> model predicted values
df['salary_hike']
Y_pred=LR.predict(df[['years_experience']])
Y_pred

###------->>>> constructing regression line between y_predict values and original values 
plt.scatter(x=df[['years_experience']],y=df['salary_hike'],color='red')
plt.scatter(x=df[['years_experience']],y=Y_pred,color='blue')
plt.plot(df[['years_experience']],Y_pred,color='black')
plt.xlabel('years of experience')
plt.ylabel('salary_hike')
plt.show()

###------>>>> metrics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(df['salary_hike'],Y_pred)
R2 = r2_score(df['salary_hike'],Y_pred)
print("Mean squared Error: ", mse.round(3))
print("Root Mean squared Error: ", np.sqrt(mse).round(3))
print("R square: ", R2.round(3))

###------> model prediction
K1= np.array([[10]])
LR.predict(K1)
























