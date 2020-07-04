
import pandas as pd
import numpy as np
import os
#cwd = os.getcwd()


#data pre-processing

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values


#filling the missing data

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values=np.nan, strategy='mean',verbose = 0)
missingvalues = missingvalues.fit(X[:,1:3])
X[:,1:3] = missingvalues.transform(X[:,1:3])

#encoding

# X data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(),[0])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X),dtype = np.float)

# Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

#spliting dataset into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)