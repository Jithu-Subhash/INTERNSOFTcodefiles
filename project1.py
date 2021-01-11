#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#reading the data from ur files

data = pd.read_csv('advertising.csv')
print(data.head())


#Visualization

fig , axs = plt.subplots(1,3,sharey= True)
data.plot(kind='scatter', x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter', x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter', x='Newspaper',y='Sales',ax=axs[2])


#Creating x&y for Linear Regression

feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

#Importing Lenear Regression Algo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)
result = 6.97 + (0.055*(50))
print(result)


#Create a dataframe with min and maxvalue of the table

X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(X_new.head())

preds = lr.predict(X_new)
print(preds)

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth= 3)

import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
print(lm.conf_int())

#Finding the probability values
print(lm.pvalues)

#Finding R-Squared values
print(lm.rsquared)

#MultiLenear Regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

lr = LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
print(lm.conf_int())
print(lm.summary())


