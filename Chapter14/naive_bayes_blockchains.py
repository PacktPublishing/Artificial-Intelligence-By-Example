#Naive Bayes applied to Blockchains
#Built with Google Translation tools
#Copyright 2018 Denis Rothman MIT License. See LICENSE.

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('data_BC.csv')

print("Blocks of the Blockchain")
print (df.head())

# Prepare the training set
X = df.loc[:,'DAY':'BLOCKS']
Y = df.loc[:,'DEMAND']

#Choose the class
clfG = GaussianNB()

# Train the model
clfG.fit(X,Y)

# Predict with the model(return the class)
print("Blocks for the prediction of the A-F blockchain")

blocks=[[12,2345,12],
        [13,2034,50],
        [25,7789,4],
        [27,6789,4]]

print(blocks)

prediction = clfG.predict(blocks)

for i in range(4):
    print("Block #",i+1," Gauss Naive Bayes Prediction:",prediction[i])


