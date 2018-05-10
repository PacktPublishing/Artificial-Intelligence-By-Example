import math
#FEEDFORWARD NEURAL NETWORK(FNN) WITH BACK PROPAGATION SOLUTION FOR XOR
#The layers are built from scratch with no deep learning library
#Copyright 2018 Denis Rothman MIT License. See LICENSE.

result=[0,0,0,0]      #trained result
train=4               #dataset size to train
training_step=0.05    # At 0.05 the network is trained in 10 steps and if set to 0.5, the network is trained in 2 steps. It a simple logic problem not a difficult numerical computation

#II hidden layer 1 and its ouput
def hidden_layer_y(epoch,x1,x2,w1,w2,w3,w4,b1,b2,pred,result):
    h1=(x1*w1)+(x2*w4)       #II.A.weight of hidden neuron h1
    h2=(x2*w3)+(x1*w2)       #II.B.weight of hidden neuron h2
  #III.threshold I,a hidden layer 2 with bias
    if(h1>=1):h1=1;
    if(h1<1):h1=0;   
    if(h2>=1):h2=1
    if(h2<1):h2=0
    
    h1= h1 * -b1
    h2= h2 * b2
    print(h1,h2)
    

#IV. threshold II and OUTPUT y
    y=h1+h2
    if(y<1 and pred>=0 and pred<2):
       result[pred]=1
 
    if(y>=1 and pred>=2 and pred<4):
       result[pred]=1


#I Feed Forward and backprogation                           
for epoch in range(50):
            if(epoch<1):
               w1=0.5;w2=0.5;b1=0.5
            w3=w2;w4=w1;b2=b1
            #I.A forward propagation on epoch 1 and IV.backpropation starting epoch 2
            for t in range (4):
                if(t==0):x1 = 1;x2 = 1;pred=0
                if(t==1):x1 = 0;x2 = 0;pred=1
                if(t==2):x1 = 1;x2 = 0;pred=2
                if(t==3):x1 = 1;x2 = 0;pred=3
                #forward propagation on epoch 1
                hidden_layer_y(epoch,x1,x2,w1,w2,w3,w4,b1,b2,pred,result)

            #Epochs information and convergence    
            print("epoch:",epoch,"optimization",round(train-sum(result)),"w1:",round(w1,4),"w2:",round(w2,4),"w3:",round(w3,4),"w4:",round(w4,4),"b1:",round(-b1,4),"b2:",round(b2,4))
            
            convergence=sum(result)-train #estimating the direction of the slope
            if(convergence>=-0.00000001): break
            

            #IV BACKWORD PROPAGATION if convergence not satisfied, backprogation is initiated
            #Weight
            
            #Using a variant of a Leaky Rectified Linear Unit(ReLu) for gradient descent
            if(convergence<0):w2+=training_step;b1=w2    # a simple step increasing value based on the estimator of convergence     
            result[0]=0;result[1]=0;result[2]=0;result[3]=0
            

                  
                  
             
                        
                       
                            
           

