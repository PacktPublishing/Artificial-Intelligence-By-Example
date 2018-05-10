import random
#FEEDFORWARD NEURAL NETWORK(FNN) WITH BACK PROPAGATION SOLUTION FOR XOR
#Layers are built from scratch with no deep learning library.
#Contains a customer order data generator (see Chapter 4 Artificial Intelligence by Example)
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
result=[0]      #result

#II hidden layer 1 and its ouput
def hidden_layer_y(x1,x2,w1,w2,w3,w4,b1,b2,result):
    h1=(x1*w1)+(x2*w4)       #II.A.weight of hidden neuron h1
    h2=(x2*w3)+(x1*w2)       #II.B.weight of hidden neuron h2

#III.threshold I,a hidden layer 2 with bias
    if(h1>=1):h1=1;
    if(h1<1):h1=0;   
    if(h2>=1):h2=1
    if(h2<1):h2=0

    h1= h1 * -b1
    h2= h2 * b2

#IV. threshold II and OUTPUT y
    y=h1+h2
    if(y<1):
       result[0]=0
 
    if(y>=1):
       result[0]=1


#I Feed Forward
subsets=0
for element in range(1000000):
            w1=0.5;w2=1;b1=1
            w3=w2;w4=w1;b2=b1
            s1=random.randint(1,500000)#choice in one set s1 
            s2=random.randint(1,500000)#choice in one set s2
            x1=random.randint(0, 1)#property of choice:size smaller=0
            x2=random.randint(0, 1)#property of choice :size bigger=1
            hidden_layer_y(x1,x2,w1,w2,w3,w4,b1,b2,result)
            if(result[0]>0):
                subsets+=1
                print("Subset:",subsets,"size subset #",x1," and ","size subset #",x2," result:",result[0],"order #"," and ",s1,"order #",s2)
            if(subsets>=8333):
               break
            result[0]=0
            
            

                  
             
                        
                       
                            
           

