import numpy as np
import pandas as pd
df=pd.read_csv("D:/data science/assignments_csv/Q7.csv")
df
####-----------------> Points
df['Points'].mean()
df['Points'].median()
df['Points'].mode()
df['Points'].var()
df['Points'].std()
Range=df['Points'].max()-df['Points'].min()
Range
#####----------------> Score
df['Score'].mean()
df['Score'].median()
df['Score'].mode()
df['Score'].var()
df['Score'].std()
Range=df['Score'].max()-df['Score'].min()
Range
####----------------> Weigh
df['Weigh'].mean()
df['Weigh'].median()
df['Weigh'].mode()
df['Weigh'].var()
df['Weigh'].std()
Range=df['Weigh'].max()-df['Weigh'].min()
Range

