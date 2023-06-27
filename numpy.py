"""
Created on Wed May 31 12:45:16 2023

NUMPY --> Numerical python
"""

import numpy as np
x1 = np.array([24,28,30,34,26])
x1
x1.ndim

data =  np.array([[24,85],[28,75],[30,78],[34,85],[26,88]])
data
data.shape
data.ndim

175,183
140,135,165

data =  np.array([[24,85,175,143],
                  [28,75,183,135],
                  [30,78,180,100],
                  [34,85,165,120],
                  [26,88,168,138]])
data

# acces the colums
  #Rows, columns
#data[start:end-1 ,start:end-1]
data[0:3 ,:]
data[: ,0:2]

# average of all the patient weight column
data[: ,1:2].min()
data[: ,1:2].max()
data[: ,1:2].mean()
data[: ,1:2].std()
data[: ,1:2].var()
















