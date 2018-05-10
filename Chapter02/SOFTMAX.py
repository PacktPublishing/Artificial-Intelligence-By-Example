# Softmax function : normalized exponential function
# Copyright 2018 Denis Rothman MIT License. See LICENSE.

import math
import numpy as np

# y is the vector of the scores of the lv vector in the warehouse example.
y = [0.0002, 0.2, 0.9,0.0001,0.4,0.6]
print('0.Vector to be normalized',y)

#Version 1 : Explicitly writing the softmax function for this case
y_exp = [math.exp(i) for i in y]
print("1",[i for i in y_exp])
print("2",[round(i, 2) for i in y_exp])
sum_exp_yi = sum(y_exp)
print("3",round(sum_exp_yi, 2))
print("4",[round(i) for i in y_exp])
softmax = [round(i / sum_exp_yi, 3) for i in y_exp]
print("5,",softmax)

#Version 2 : Explicitly but with no comments
y_exp = [math.exp(i) for i in y]
sum_exp_yi = sum(y_exp)
softmax = [round(i / sum_exp_yi, 3) for i in y_exp]
print("6, Normalized vector",softmax)

#Version 3: Using a function in a 2 line code instead of 3 lines
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print("7A Normalized vector",softmax(y))
print("7B Sum of the normalize vector",sum(softmax(y)))

ohot=max(softmax(y))
ohotv=softmax(y)

print("7C.Finding the highest value in the normalized y vector : ",ohot)
print("7D.For One-Hot function, the highest value will be rounded to 1 then set all the other values of the vector to 0: ")
for onehot in range(6):
    if(ohotv[onehot]<ohot):
        ohotv[onehot]=0
    if(ohotv[onehot]>=ohot):
        ohotv[onehot]=1
print("This is a vector that is an output of a one-hot function on a softmax vector")
print(ohotv)
