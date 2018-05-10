# Markov Decision Process (MDP) - The Bellman equations adapted to
# Reinforcement Learning with the Q action-value(reward) function.
# Convergence function added to the version in 01-MDP
# Copyright 2018 Denis Rothman MIT License. See LICENSE.


import numpy as ql
import math


# Markov Decision Process (MDP) - Bellman's equations adapted to
# Reinforcement Learning
# R is The Reward Matrix for each location in a warehouse (or any other problem)
R = ql.matrix([ [-1,-1,-1,-1,0,-1],
		[-1,-1,-1,0,-1,0],
		[-1,-1,100,0,-1,-1],
		[-1,0,100,-1,0,-1],
		[0,-1,-1,0,-1,-1],
		[-1,0,-1,-1,-1,-1] ])

# Q is the Learning Matrix in which rewards will be learned/stored
Q = ql.matrix(ql.zeros([6,6]))
Q_C=Q

# Gamma : It's a form of penalty or uncertainty for learning
# If the value is 1 , the rewards would be too high.
# This way the system knows it is learning.
gamma = 0.8
#ceg = goal of this training session to limit the number of iterations
# ceg is optional but can save time if and only if you know the result you
# want the system to be trained for.
ceg=3992

# agent_s_state. The agent the name of the system calculating
# s is the state the agent is going from and s' the state it's going to
# this state can be random or it can be chosen as long as the rest of the choices
# are not determined. Randomness is part of this stochastic process
agent_s_state = 1

# The possible "a" actions when the agent is in a given sate
def possible_actions(state):
    current_state_row = R[state,]
    possible_act = ql.where(current_state_row >= 0)[1]
    return possible_act

# Get available actions in the current state
PossibleAction = possible_actions(agent_s_state) 

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def ActionChoice(available_actions_range):
    next_action = int(ql.random.choice(PossibleAction,1))
    return next_action

# Sample next action to be performed
action = ActionChoice(PossibleAction)

# Maximum Convergence value 
sum=0

# A version of Bellman's equation for Q Learning
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
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue

# Rewarding Q matrix
reward(agent_s_state,action,gamma)


# Leraning over n iterations depending on the convergence of the system
# A convergence function can replace the systematic repeating of the process
# by comparing the sum of the Q matrix to that of Q matrix n-1 in the
# previous episode
for i in range(50000):
    current_state = ql.random.randint(0, int(Q.shape[0]))
    PossibleAction = possible_actions(current_state)
    action = ActionChoice(PossibleAction)
    reward(current_state,action,gamma)
    if Q.sum()>0:
     #print("convergent episode:",i,"Q.Sum",Q.sum(),"numerical convergent value e-1:",Q.sum()-sum)       
     #print("convergent episode:",i,"numerical convergent value:",ceg-Q.sum())
     CEV=-(math.log(Q.sum())-math.log(ceg))
     print("convergent episode:",i,"numerical convergent value:",CEV)
     sum=Q.sum()
     if(Q.sum()-3992==0):
       print("Final convergent episode:",i,"numerical convergent value:",ceg-Q.sum())
       break; #break on average (the process is random) before 50000
     
    

# Displaying Q before the norm of Q phase

print("Q before norm Q")
print(Q)

print("Sum Q",Q.sum())

# Norm of Q
print("Normed Q :")
print(Q/ql.max(Q)*100)
