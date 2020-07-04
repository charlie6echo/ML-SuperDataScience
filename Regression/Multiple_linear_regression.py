


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1] .values
y = dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#remove dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
X_opt = np.array(X[:,[0,1,2,4,5]],dtype=float)
regressor_ols = sm.OLS(endog = y,exog = X_opt ).fit()
regressor_ols.summary()
X_opt = np.array(X[:,[0,1,2,4,5]],dtype=float)
regressor_ols = sm.OLS(endog = y,exog = X_opt ).fit()
regressor_ols.summary()
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#plt.scatter(X_train,y_train,color = 'red')
#plt.plot()