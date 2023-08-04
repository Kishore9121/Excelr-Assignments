
from scipy import stats 
X_mean=200
X_std=30

######-------->94% confidence interval
df_ci=stats.norm.interval(0.94,loc=X_mean,scale=X_std)
print('94% confident lies between',df_ci)

######-------->98% confidence interval
df_ci=stats.norm.interval(0.98,loc=X_mean,scale=X_std)
print('98% confident lies between',df_ci)

######-------->96% confidence interval
df_ci=stats.norm.interval(0.96,loc=X_mean,scale=X_std)
print('96% confident lies between',df_ci)
