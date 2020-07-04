
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
regressor.fit(X,y)
print(type(X))
regressor.predict([[6.5]])
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid))
plt.title('random forest')
plt.xlabel('positon')
plt.ylabel('salary')
plt.show()
