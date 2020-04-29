# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:44:00 2020

@author: MrROBOT
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#data
X = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])


#ploting the data
plt.scatter(X[:,0],X[:,1],label ='True positive')
plt.show()

#model selection
kmeans = KMeans(n_clusters =3)

#fitting the model
kmeans.fit(X)

#printing centroid values
print(kmeans.cluster_centers_)
#printing predicted values
print(kmeans.labels_)

#plotting the data based on their predicted clusters
plt.scatter(X[:,0],X[:,1],c = kmeans.labels_, cmap = 'rainbow')
plt.show()
