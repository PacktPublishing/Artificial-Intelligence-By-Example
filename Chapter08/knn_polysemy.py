#K-means clustering 
#Build with Sklearn
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Importthe persed data
df = pd.read_csv('V1.csv')

print (df.head())

# KNN Classification labels
X = df.loc[:,'broke':'shouted']
Y = df.loc[:,'class']

# Trains the model
knn = KNeighborsClassifier()
knn.fit(X,Y)

# Requesting a prediction
#broke and stopped are
#activated to see the best choice of words to fit these features.
# brock and stopped were found in the sentence to be interpreted.
# In X_DL as in X, the labels are : broke, road, stopped,shouted.
X_DL = [[9,0,9,0]] 
prediction = knn.predict(X_DL)
print ("The prediction is:",str(prediction).strip('[]'))

#Uses the same V1.txt because the parsing has
# been checked and is reliable as "dataset lexical rule base".
df = pd.read_csv('V1.csv') 
# Plotting the relation of each feature with each class
figure,(sub1,sub2,sub3,sub4)=plt.subplots(4,sharex=True,sharey=True)
plt.suptitle('k-nearest neighbors')
plt.xlabel('Feature')
plt.ylabel('Class') 
X = df.loc[:,'broke']
Y = df.loc[:,'class']
sub1.scatter(X, Y,color='blue',label='broke')
sub1.legend(loc=4, prop={'size': 5})
sub1.set_title('Polysemy')
X = df.loc[:,'road']
Y = df.loc[:,'class']
sub2.scatter(X, Y,color='green',label='road')
sub2.legend(loc=4, prop={'size': 5})
X = df.loc[:,'stopped']
Y = df.loc[:,'class']
sub3.scatter(X, Y,color='red',label='stopped')
sub3.legend(loc=4, prop={'size': 5})
X = df.loc[:,'shouted']
Y = df.loc[:,'class']
sub4.scatter(X, Y,color='black',label='shouted')
sub4.legend(loc=4, prop={'size': 5})
figure.subplots_adjust(hspace=0)
plt.show()
















