# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:04:08 2020

@author: SHUKLA
"""


# ML A-Z  Tutorials
# --Polynomial regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting poynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#plot loinear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("linear regression graphs")
plt.xlabel("level of positions")
plt.ylabel("salaries")
plt.show()

#plot polynomial regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title("linear regression graphs")
plt.xlabel("level of positions")
plt.ylabel("salaries")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting using linear regression model 
#-- use [[X]] or  np.array([X]).reshape(1, 1)  for prediction or else it will through array error
lin_reg.predict([[6.5]])

#predicting using polynomial regression model
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))