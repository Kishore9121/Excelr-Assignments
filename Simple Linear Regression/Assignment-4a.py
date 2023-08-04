###----> import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###---->import dataset
df=pd.read_csv('D:/data science/assignments_csv/delivery_time.csv')
df

###---->Renaming columns
df=df.rename({'Sorting Time':'sorting_time','Delivery Time':'delivery_time'},axis=1)
df

###----->step-2:EDA
###-----> performing scatterplot analysis 
plt.scatter(x=df['sorting_time'],y=df['delivery_time'],color='red')
plt.xlabel('sorting_time')
plt.ylabel('delivery_Time')
plt.show()

###-----> constructing box plot for verifiying outliers
df.boxplot(column='sorting_time',vert=False)
df.boxplot(column='delivery_time')

####-----> correlation analysis
df.corr()

###------> model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(df[['sorting_time']],df['delivery_time'])
LR.intercept_ # Bo
LR.coef_

###------> model predicted values
df['delivery_time']
Y_pred = LR.predict(df[['sorting_time']])

###------> constructing regrassion line between model predicted values and original values 
import matplotlib.pyplot as plt
plt.scatter(x=df[['sorting_time']],y=df['delivery_time'],color='red')
plt.scatter(x=df[['sorting_time']],y=Y_pred,color='blue')
plt.plot(df[['sorting_time']],Y_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()

###------> metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(df['delivery_time'],Y_pred)
R2 = r2_score(df['delivery_time'],Y_pred)
print("Mean squared Error: ", mse.round(3))
print("Root Mean squared Error: ", np.sqrt(mse).round(3))
print("R square: ", R2.round(3))

###------> model prediction
K1= np.array([[6]])
LR.predict(K1)

