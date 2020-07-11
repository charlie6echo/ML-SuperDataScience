#imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#data
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

#data cleaning
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]).lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
  
#creating model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.metrics import confusion_matrix
def ConfusionMat(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    accuracy = (cm[0][0] + cm[1][1])/np.sum(cm)
    return cm,accuracy

# testing each classifiaction model

#1. Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
LRCM = cm
print('Logistic Regression = ',accuracy)

#2.Decision Trees

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
DTCM = cm
print('Decision Tree = ',accuracy)

#3.Kernel SVM

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
SVMCM = cm
print('Support Vector Machine = ',accuracy)

#4. KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p =2)
classifier.fit(X_train,y_train)
classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
KNNCM = cm
print('KNN = ',accuracy)

#5.naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
NBCM = cm
print('naive Bayes = ',accuracy)

#random forest 
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier(n_estimators = 100 ,criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
classifier.predict(X_test)
cm,accuracy = ConfusionMat(y_test,y_pred)
RFCM = cm
print('Random Forest = ',accuracy)
