###########################################################################################
#### Import Libraries
import pandas as pd
import numpy as np

#### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/LabTAT.csv')
df

#### Mean
df['Laboratory 1'].mean()
df['Laboratory 2'].mean()
df['Laboratory 3'].mean()
df['Laboratory 4'].mean()

#### Histogram
df['Laboratory 1'].hist()
df['Laboratory 2'].hist()
df['Laboratory 3'].hist()
df['Laboratory 4'].hist()

#### Skewness
df['Laboratory 1'].skew()
df['Laboratory 2'].skew()
df['Laboratory 3'].skew()
df['Laboratory 4'].skew()


#### Anova Test
##--->H0:there is significance difference in average TAT among the different laboratories at 5% significance level.
##--->H1:there is no significance difference in average TAT among the different laboratories at 5% significance level.
import scipy.stats as stats
Fcalc, pvalue = stats.f_oneway(df["Laboratory 1"],df["Laboratory 2"],df["Laboratory 3"],df["Laboratory 4"])

Fcalc
pvalue

alpha=0.05
if (pvalue < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is rejected and H1 is accepted that means there is no significance
 difference in average TAT among the different laboratories at 5% significance level.'''
        










