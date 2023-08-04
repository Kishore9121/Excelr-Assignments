import pandas as pd 
df=pd.read_csv('D:/data science/assignments_csv/Q9_a.csv')
df
#####------------->skewness for speed and dist
df['speed'].skew()
df['dist'].skew()
#####-------------->kortosis() for speed and dist
df['speed'].kurtosis()
df['dist'].kurtosis()
