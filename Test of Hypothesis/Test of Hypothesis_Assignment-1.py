###########################################################################################
#### Import Libraries
import pandas as pd
import numpy as np

#### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/Cutlets.csv')
df

#### Renaming the columns
df.rename(columns={'Unit A':'Unit_A','Unit B':'Unit_B'},inplace=True)

#### Mean
df['Unit_A'].mean()
df['Unit_B'].mean()

#### Skewness
df['Unit_A'].skew()
df['Unit_B'].skew()

#### Histogram
df['Unit_A'].hist()
df['Unit_B'].hist()

#### Two sample Z-Test
##--->Null Hypothesis:There is no significance difference between diameter of the Cutlets
##--->Alternative Hypothesis:There is a significance difference between diameter of the Cutlets
from scipy import stats
zcalc ,pval = stats.ttest_ind( df["Unit_A"] , df["Unit_B"]) 

print("Zcalcualted value is ",zcalc.round(4))
print("P-value is ",pval.round(4))

if pval<0.05:
    print("reject null hypothesis, Accept Alternative hypothesis")
else:
    print("accept null hypothesis, Reject Alternative hypothesis")
    
'''Here we are accepting null hypothesis and rejecting alternative hypothesis
   that means there is no significance between diameter of the cutlets'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    