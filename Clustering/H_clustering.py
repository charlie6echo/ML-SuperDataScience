


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values 


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('dendrogram')
plt.xlabel('income')
plt.ylabel('euclidian')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage = 'ward')


yhc = hc.fit_predict(X)

plt.scatter(X[yhc == 0,0], X[yhc == 0,1], s =100 , c = 'blue' ,label = 'careful')
plt.scatter(X[yhc == 1,0], X[yhc == 1,1], s =100 , c = 'magenta' ,label = 'standard')
plt.scatter(X[yhc == 2,0], X[yhc == 2,1], s =100 , c = 'red' ,label = 'Target')
plt.scatter(X[yhc == 3,0], X[yhc == 3,1], s =100 , c = 'cyan' ,label = 'careless')
plt.scatter(X[yhc == 4,0], X[yhc == 4,1], s =100 , c = 'green' ,label = 'sensible')
plt.title('cluster of customers')
plt.xlabel('income/salary')
plt.ylabel('spending')
plt.legend()
plt.show()