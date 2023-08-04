###########################################################################################
#### Import Libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

#### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/BuyerRatio (1).csv')
df

###############################################################################
df.info()
df.describe()

#### Set the 'Gender' column as the index (optional, for better representation)
df.set_index('Observed Values', inplace=True)

#### Perform the Chi-Square test
chi2_stat, p_val, dof, expected = chi2_contingency(df)

#### Print the results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of Freedom:", dof)

#Ho--> Independency
#H1-->Dependency

alpha=0.05
if (p_val < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is accepted and H1 is rejected that means there is no significance
    dependency between Male and female buyers and similar in group.Hence they
    are Independent samples'''
