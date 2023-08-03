###########################################################################################
#### Import Libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

#### Import Dataset
df=pd.read_csv('D:/data science/assignments_csv/Costomer+OrderForm.csv')
df

#### Value counts
df['Phillippines'].value_counts()
df['Indonesia'].value_counts()
df['Malta'].value_counts()
df['India'].value_counts()

#### checking datatypes
df.info()
df.describe()

# Create a contingency table using pd.crosstab
contingency_table = pd.crosstab(df['Phillippines'], [df['Indonesia'], df['Malta'], df['India']])

# Perform the chi-square test on the contingency table
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)

#Ho--> Independency
#H1-->Dependency

alpha=0.05
if (p_val < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is accepted and H1 is rejected that means there is no significance
    dependency between countries and similar in groups.Hence they
    are Independent'''

