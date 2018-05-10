#Rectified Linear Unit(ReLU)
#Built with Numpy
#Copyright 2018 Denis Rothman MIT License. READ LICENSE.
import numpy as np 
nx=-3
px=5

def relu(x):
    if(x<=0):ReLU=0
    if(x>0):ReLU=x
    return ReLU

def f(x):
    vfx=np.maximum(0.1,x)
    return vfx

def lrelu(x):
    if(x<0):lReLU=0.01
    if(x>0):lReLU=x
    return lReLU


print("negative x=",nx,"positive x=",px)
print("ReLU nx=",relu(nx))
print("ReLU px=",relu(px))
print("Leaky ReLU nx=",lrelu(nx))
print("f(nx) ReLu=",f(nx))
print("f(px) ReLu=",f(px))
print("f(0):",f(0))
