##############################################################################
##### Import Libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules

##### Importing dataset
df = pd.read_csv('D:/data science/assignments_csv/my_movies.csv')
df.columns
df.info()
df['V1'].value_counts()
df['V2'].value_counts()
df['V3'].value_counts()
df['V4'].value_counts()
df['V5'].value_counts()

df.drop(columns=['V1','V2','V3','V4','V5'],inplace=True)
df
##############################################################################
##### Apriori Algorithm 
##### min_support=0.2
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets
frequent_itemsets.shape

##### Association Rules
#### min_threshold=0.7
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.shape
list(rules)

#### shorting the rules in ascending 
rules.sort_values('lift',ascending = False)

#### shorting the rules in ascending with 10 
rules.sort_values('lift',ascending = False)[0:10]

#### Lift value greater than 1
rules[rules.lift>1]

#### Histogram for support , cinfidence and lift
rules[['support','confidence','lift']].hist()

#### Scatter plot between support and confidence 
import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()

#### scatter plot between support and confidence using seborn
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()

##############################################################################
##### Apriori Algorithm 
##### min_support=0.1
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets
frequent_itemsets.shape

##### Association Rules
#### min_threshold=0.5
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules
rules.shape
list(rules)

#### shorting the rules in ascending 
rules.sort_values('lift',ascending = False)

#### shorting the rules in ascending with 20 
rules.sort_values('lift',ascending = False)[0:20]

#### Lift value less than 2
rules[rules.lift<2]

#### Histogram for support , cinfidence and lift
rules[['support','confidence','lift']].hist()

#### Scatter plot between support and confidence 
import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()

#### scatter plot between support and confidence using seborn
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()

''' For apriori i tried with two different cases and plotted different graphs
    Case-1: Minimum_support=0.2,Minimum_threshold=0.7 and lift greater than 1
    Case-2: Minimum_support=0.01,Minimum_threshold=0.5 and lift less than 2'''






