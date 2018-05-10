#K-means clustering - Mini-Batch-Shuffling
#Build with Sklearn
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
from sklearn.cluster import KMeans  
import pandas as pd
from matplotlib import pyplot as plt
from random import randint
import numpy as np

#I.The training Dataset 
dataset = pd.read_csv('data.csv')
print (dataset.head())
print("initial order")
print(dataset)

sn=4999
shuffled_dataset=np.zeros(shape=(sn,2))
for i in range (sn):
    shuffled_dataset[i][0]=dataset.iloc[i,0]
    shuffled_dataset[i][1]=dataset.iloc[i,1]

print("shuffled")
print(shuffled_dataset)

    



'''Output of print(dataset)
      Distance  location
0           80        53
1           18         8
2           55        38
...
'''

n=1000
dataset1=np.zeros(shape=(n,2))
for i in range (n):
    dataset1[i][0]=shuffled_dataset[i,0]
    dataset1[i][1]=shuffled_dataset[i,1]



'''
li=0
for i in range (n):
    j=randint(0,4999)
    dataset1[li][0]=dataset.iloc[j,0]
    dataset1[li][1]=dataset.iloc[j,1]
    li+=1
    
'''
#II.Hyperparameters
# Features = 2
k = 6
kmeans = KMeans(n_clusters=k)

#III.K-means clustering algorithm
kmeans = kmeans.fit(dataset1)         #Computing k-means clustering
gcenters = kmeans.cluster_centers_   # the geometric centers or centroids
print("The geometric centers or centroids:")
print(gcenters)

'''Ouput of centroid coordinates

The geometric centers or centroids:

Monte Carlo philosophy:

MC[[ 19.7877095   16.40782123]
  [ 21.38965517  15.04597701]]

MC [ 99.87603306  81.1322314 ]
   [ 96.06151645  84.57939914]

MC[ 31.29139073  72.64900662]]
  [ 32.12590799  54.84866828]

MC [ 61.54891304  49.875     ]
   [ 68.84578885  55.63226572]


MC [ 63.86206897  84.20689655]
   [ 45.24736842  23.65263158]


Complete dataset:

[[ 48.7986755   85.76688742]
 [ 48.44532803  24.4333996 ]
'''


#IV.Defining the Result labels 
labels = kmeans.labels_
colors = ['blue','red','green','black','yellow','brown','orange']


#V.Displaying the results : datapoints and clusters
y = 0
for x in labels:
    plt.scatter(dataset1[y,0], dataset1[y,1],color=colors[x])
    y+=1       
for x in range(k):
    lines = plt.plot(gcenters[x,0],gcenters[x,1],'kx')    

title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('Distance')
plt.ylabel('Location')
plt.show()

#VI.Test dataset and prediction
x_test = [[40.0,67],[20.0,61],[90.0,90],
          [50.0,54],[20.0,80],[90.0,60]]
prediction = kmeans.predict(x_test)
print("The predictions:")
print (prediction)
'''
Output of the cluster number of each example
[3 3 2 3 3 4]
'''

