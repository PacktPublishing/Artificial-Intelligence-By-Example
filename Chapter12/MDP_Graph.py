# Copyright 2018 Denis Rothman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# A type of DQN with a CNN, a CRL optimization function and an MDP function
# activated by an input stream of frames

import numpy as ql
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random
import math
import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import load_model
import scipy.misc      
from PIL import Image

# I.Markov Decision Process (MDP) - The Bellman equations adapted to
# Reinforcement Learning with the Q action-value(reward) function.

L=['A','B','C','D','E','F']
#initial weight of nodes(vertices) with no edges directions (undirected graph)
W=[0,0,0,0,0,0]             
# R is The Reward Matrix for each state built on the physical graph
# Ri is a memory of this initial state: no rewards and undirected
R = ql.matrix([ [0,0,0,0,1,0],
		[0,0,0,1,0,1],
		[0,0,0,1,0,0],
		[0,1,1,0,1,0],
		[1,0,0,1,0,0],
		[0,1,0,0,0,0] ])

Ri = ql.matrix([ [0,0,0,0,1,0],
		[0,0,0,1,0,1],
		[0,0,0,1,0,0],
		[0,1,1,0,1,0],
		[1,0,0,1,0,0],
		[0,1,0,0,0,0] ])


# Q is the Learning Matrix in which rewards will be learned/stored
Q = ql.matrix(ql.zeros([6,6]))


#II. Convolutional Neural Network (CNN)
#loads,traffic,food processing
A=['dataset_O/','dataset_traffic/','dataset/']
MS1=['loaded','jammed','productive']
MS2=['unloaded','change','gap']

display=1     #display images     
scenario=2    #reference to A,MS1,MS2
directory=A[scenario] #transfer learning parameter (choice of images)
CRLMN=1       # concept learning
print("Classifier frame directory",directory)

# Learning over n iterations depending on the convergence of the system
# A convergence function can replace the systematic repeating of the process
# by comparing the sum of the Q matrix to that of Q matrix n-1 in the
# previous episode

#____________________LOAD MODEL____________________________
json_file = open(directory+'model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
#___________________ load weights into new model
loaded_model.load_weights(directory+"model/model.h5")
print("Strategy model loaded from training repository.")
# __________________compile loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



#___________________IDENTIFY IMAGE FUNCTION_____________________

def identify(target_image,e):
    filename = target_image
    original = load_img(filename, target_size=(64, 64))
    #print('PIL image size',original.size)
    fn=str(e)
    if(display==1):
        plt.subplot(111)
        plt.imshow(original)
        plt.title('STEP 1: A web cam freezes frame # '+ fn + ' of a conveyor belt.'+ '\n' + 'The frame then becomes the input of a trained Keras-Tensorflow CNN.' +'\n'+'The output will classify the frame as well loaded or containing a gap.',fontname='Arial', fontsize=10)
        #plt.text(0.1,2, "The frame is the input of a trained CNN")
        plt.show(block=False)
        time.sleep(5)
        plt.close()
    numpy_image = img_to_array(original)
    arrayresized = scipy.misc.imresize(numpy_image, (64,64))    
    #print('Resized',arrayresized)
    inputarray = arrayresized[ql.newaxis,...] # extra dimension to fit model 
#___________________PREDICTION___________________________
    prediction1 = loaded_model.predict_proba(inputarray)
    prediction2 = loaded_model.predict(inputarray)
    print("image",target_image,"predict_probability:",prediction1,"prediction:",prediction2)
    return prediction1

#I._________________MARKOV DECISION PROCESS_______________

#Logistic Signmoid function to squash the weights 
def logistic_sigmoid(w):
  return 1 / (1 + math.exp(-w))

# Gamma : It's a form of penalty or uncertainty for learning
# If the value is 1 , the rewards would be too high.
# This way the system knows it is learning.
gamma = 0.8

# agent_s_state. The agent the name of the system calculating
# s is the state the agent is going from and s' the state it's going to
# this state can be random or it can be chosen as long as the rest of the choices
# are not determined. Randomness is part of this stochastic process
agent_s_state = 1

# The possible "a" actions when the agent is in a given state
def possible_actions(state):
    current_state_row = R[state,]
    possible_act = ql.where(current_state_row >0)[1]
    return possible_act

# Get available actions in the current state
PossibleAction = possible_actions(agent_s_state)

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def ActionChoice(available_actions_range):
    if(sum(PossibleAction)>0):
        next_action = int(ql.random.choice(PossibleAction,1))
    if(sum(PossibleAction)<=0):
        next_action = int(ql.random.choice(5,1))
    return next_action

# Sample next action to be performed
action = ActionChoice(PossibleAction)

# A version of Bellman's equation for reinforcement learning using the Q function
# This reinforcement algorithm is a memoryless process
# The transition function T from one state to another
# is not in the equation below.  T is done by the random choice above

def reward(current_state, action, gamma):
    Max_State = ql.where(Q[action,] == ql.max(Q[action,]))[1]

    if Max_State.shape[0] > 1:
        Max_State = int(ql.random.choice(Max_State, size = 1))
    else:
        Max_State = int(Max_State)
    MaxValue = Q[action, Max_State]
    
    # Bellman's MDP based Q function
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue

# Rewarding Q matrix
reward(agent_s_state,action,gamma)


#I CNN AND II MDP form a type of DQN (Deep Q Network):
#A. frame of a stream (video or other) goes through a trained CNN
#B. an optimizing function guides the MDP to an efficient target
#C. The MDP optimizes the process
def CRLMM(Q,lr,e):
    # Displaying Q before the norm of Q phase
    print("Q  :")
    print(Q)
    # Norm of Q
    print("Normed Q :")
    print(Q/ql.max(Q)*100)
    #Graph structure
    RL=['','','','','','']
    RN=[0,0,0,0,0,0]
    print("State of frame :",lr,L[lr])
    for i in range(6):
        maxw=0
        for j in range(6):
            W[j]+=logistic_sigmoid(Q[i,j])
            if(Q[i,j]>maxw):
                RL[i]=L[j]
                RN[i]=Q[i,j]
                maxw=Q[i,j]
                print(i,L[i],RL[i],RN[i])
    status=random.randint(0,1)
    if(status==0):
      #Add frame from video stream (connect to webcam)
      s=identify(directory+'classify/img1.jpg',e)
    if(status==1):
      #Add frame from video stream (connect to webcam)
      s=identify(directory+'classify/img2.jpg',e)
    s1=int(s[0])
    if (int(s1)==0):
      print('Classified in class A')
      print(MS1[scenario])
      print('Seeking...')
    if (int(s1)==1):
      print('Classified in class B')
      print(MS2[scenario])
    return s1

#Displaying the Weights    
def MDP_CRL_graph(W,t):
  Y=[1,2,3,4,5,6]
  h=int(round(max(W),3))+2

  fig,ax=plt.subplots()
  plt.bar(Y,W)
  plt.xticks(range(0,7))
  plt.yticks(range(0,h))
 
  plt.xticks(Y,('A','B','C','D','E','F'))
  plt.title('STEP 3: The Vertice Weights are UPDATED after the MDP for: '+L[t],fontname='Arial', fontsize=10)
  
  plt.show(block=False)
  time.sleep(5)
  plt.close()

#Displaying the MDP graph
def MDP_GRAPH(x,e):
  verts1 = [
      (1., 0.8),  # F
      (0.6, 0.7), # B
      (0.4, 0.3), # D
      (0.8, 0.),  # C
      ]

  codes1 = [Path.MOVETO,
           Path.MOVETO,
           Path.MOVETO,
           Path.MOVETO,
           ]

  fig = plt.figure()
  ax = fig.add_subplot(111)

  path1 = Path(verts1, codes1)
  patch1 = patches.PathPatch(path1, facecolor='none', lw=2)
  ax.add_patch(patch1)

  verts2 = [
      (0.2, 1.),  # A
      (0., 0.),   # E
      (0.4, 0.3), # D
      ]

  codes2 = [Path.MOVETO,
           Path.MOVETO,
           Path.MOVETO,
           ]

  path2 = Path(verts2, codes2)
  patch2 = patches.PathPatch(path2, facecolor='none', lw=2)
  ax.add_patch(patch2)

  tvc1='green'
  if(x==1 or x==2 or x==5 or x==3):
    tvc1='red'
    
  tvc2='green'   
  if(x==0 or x==4):
    tvc2='red'

  vtit=L[x]
    
  xs1, ys1 = zip(*verts1)
  ax.plot(xs1, ys1, 'o--', lw=2, color=tvc1, ms=10)

  xs2, ys2 = zip(*verts2)
  ax.plot(xs2, ys2, 'o--', lw=2, color=tvc2, ms=10)

  ax.text(-0.05, -0.05, 'E')
  ax.text(0.15, 1.05, 'A')
  ax.text(1.05, 0.85, 'F')
  ax.text(0.85, -0.05, 'C')
  ax.text(0.6, 0.75, 'B')
  ax.text(0.4, 0.35, 'D')

  fn=str(e)
  plt.title('STEP 2'+' Frame#' + fn+ ': CRL-MDP OPTIMIEZS VERTICE  '+ vtit + '\n'+ 'The optimizer chose the main available vertice in the red network)'+ '\n'+ 'MDP will spread the load out to other vertices in the red network',fontname='Arial', fontsize=10)

  ax.set_xlim(-0.1, 1.1)
  ax.set_ylim(-0.1, 1.1)

  plt.show(block=False)
  time.sleep(5)
  plt.close()


# The DQN : a trained CNN + GAP optimization + MDP repeated as long as the stream inputs data
# The episodes are the input frames 

episodes=11   #input frames
crlmm=1000    #waiting state
#input_output_frequency : output every n frames/ retained memory
oif=10
#input_output_rate p% (memory retained)
oir=0.2
fc=0 #frequencey counter : memory output
for e in range(episodes):
  print("episode:frame #",e)
  fc=fc+1
  #memory management : lambda output
  if(fc>=10):
    for fci in range(6):
      W[fci]=W[fci]*oir
      fc=0
      print("OUTPUT OPERATION - MEMORY UPDATED FOR ",L[fci]," ",oir,"% retained")
  #target recaclulated for each episode
  lr=0
  #first episode is random
  if(e==0):
    lr=random.randint(0,5)
  #if episode>1, then min or max
  #BEGINNING OF GAP OPTIMIZATION PROCESS : a key real-time function
  #in this model, min is the target but max could fit another model
  if(e>0):
    lr=0
    minw=10000000
    maxw=-1
    #no G => Finding the largest gap (most loaded resource or a distance)
    if(crlmm==0):
      for wi in range(3):
        op=random.randint(0,5)
        if(W[op]<minw):
          lr=op;minw=W[op]
    #G =>  Finding the smallest gap (a least loaded resource or a distance)
    if(crlmm==1):
      for wi in range(3):
          op=random.randint(0,5)
          if(W[op]>maxw):
           lr=op;maxw=W[op]
         
  print("LR TARGET STATE MDP number and letter:",lr,L[lr])
  #initial reward matrix set again
  for ei in range(6):
    for ej in range(6):
      Q[ei,ej]=0
      if(ei !=lr):
        R[ei,ej]=Ri[ei,ej]
      if(ei ==lr):
        R[ei,ej]=0 #to target, not from
        
  #target reward updated by the result of the real-time optimization process
  #
  #no G
  rew=100
  #G
  if(crlmm==1):
    rew=50
  R[lr,lr]=rew
  print("Initial Reward matrix withe vertice locations:",R)
#_____END OF GAP OPTIMIZATION PROCESS
  agent_s_state = 1 #can be random
  # Get available actions in the current state
  PossibleAction = possible_actions(agent_s_state)
  # Sample next action to be performed
  action = ActionChoice(PossibleAction)
  # Rewarding Q matrix
  reward(agent_s_state,action,gamma)

  for i in range(50000):
    current_state = ql.random.randint(0, int(Q.shape[0]))
    PossibleAction = possible_actions(current_state)
    action = ActionChoice(PossibleAction)
    reward(current_state,action,gamma)

  crlmm=CRLMM(Q,lr,e)
  print("GAP =0 or GAP =1 status: ",crlmm)
  MDP_GRAPH(lr,e)
  print("Vertice Weights",W)
  MDP_CRL_graph(W,lr)
    
