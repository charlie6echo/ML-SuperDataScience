
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values 

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
    

kmeans = KMeans(n_clusters = 5,init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0 )
ykmeans = kmeans.fit_predict(X)

plt.scatter(X[ykmeans == 0,0], X[ykmeans == 0,1], s = 100, c = 'magenta', label = 'standard')
plt.scatter(X[ykmeans == 1,0], X[ykmeans == 1,1], s = 100, c = 'cyan' ,label = 'careless')
plt.scatter(X[ykmeans == 2,0], X[ykmeans == 2,1], s = 100, c = 'red' ,label = 'Target')
plt.scatter(X[ykmeans == 3,0], X[ykmeans == 3,1], s = 100, c = 'green', label = 'sensible')
plt.scatter(X[ykmeans == 4,0], X[ykmeans == 4,1], s = 100, c = 'blue' ,label = 'careful')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow' ,label = 'centriods')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()