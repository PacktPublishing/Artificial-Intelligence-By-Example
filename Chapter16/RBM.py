#Cognitive NPL (Natural Language Processing) Chabot
#Denis Rothman
#Personality Profiling with a Restricted Botzmannm Machine (RBM)
#Only the Initial RBM : https://github.com/echen/restricted-boltzmann-machines
#The concepts and the rest of the code were designed and writtent by Denis Rothman
#Sentiment Analysis with Textblob
#Copyright 2018 Denis Rothman MIT License. READ LICENSE.
from __future__ import print_function
import numpy as np
from textblob import TextBlob
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time

class RBM:
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)).
    # Standard initialization  the weights with mean 0 and standard deviation 0.1. 
    #Starts with random state 
    np_rng = np.random.RandomState(1234)
    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_associations = np.dot(data.T, pos_hidden_probs)
      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)
    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # hidden_states[:,0] = 1
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    num_examples = data.shape[0]
    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    visible_states = np.ones((num_examples, self.num_visible + 1))
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)
    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    # Create a matrix, where each row is to be a sample of of the visible units 
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    for i in range(1, num_samples):
      visible = samples[i-1,:]
      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1
      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states
    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 6, num_hidden = 2)
  training_data = np.array([[1,1,0,0,1,1],
                            [1,1,0,1,1,0],
                            [1,1,1,0,0,1],
                            [1,1,0,1,1,0],
                            [1,1,0,0,1,0],
                            [1,1,1,0,1,0]])

  F=["love","happiness","family","horizons","action","violence"]
  print(" A Restricted Boltzmann Machine(RBM)","\n","applied to profiling a person name X","\n","based on the movie ratings of X.","\n")
  print("The input data represents the features to be trained to learn about person X.")
  print("\n","Each colum represents a feature of X's potential pesonality and tastes.")
  print(F,"\n")
  print(" Each line is a movie X watched containing those 6 features")
  print(" and for which X gave a 5 star rating.","\n")
  
  print(training_data)

  print("\n")
  m=input("Press ENTER if you agree to start learning about X")
  r.train(training_data, max_epochs = 1000)
  print("\n","The weights of the features have been trained for person X.","\n","The first line is the bias and examine column 2 and 3","\n","The following 6 lines are X's features.","\n")
  print("Weights:")
  print(r.weights)
  print("\n","The following array is a reminder of the features of X.")
  print(" The columns are the potential features of X.","\n", "The lines are the movies highly rated by X")
  print(F,"\n")
  print(training_data)
  print("\n")
  print("The results are only experimental results.","\n")
  ct=input("Press ENTER if you agree to see the profile of X:")
  
  
  for w in range(7):
      if(w>0):
          W=print(F[w-1],":",r.weights[w,1]+r.weights[w,2])
          
  print("\n")
  print("A value>0 is positive, close to 0 slightly positive")
  print("A value<0 is negative, close to 0 slightly negative","\n")
  
        
  m=input("Press ENTER if you agree to learn more about X.")
  print("The AI program will now enter social networks")
  print("to scan X's social media profile.","\n")
  print("First words(Tweets, posts, messages on Facebook and much more) will be analyzed then images","\n")
  m=input("Press ENTER if you agree to scan X's social networking...")
  print("Sentiment Analysis with TextBlob","\n")
  print("The following sentences were scanned on X's social networks","\n")



mytext=input("Press ENTER to Continue")

print("A value>0 is positive, close to 0 slightly positive")
print("A value<0 is negative, close to 0 slightly negative","\n")

myview=TextBlob("I hate movie 1. It was too violent ")
print(myview,":","\n",myview.sentiment,"\n")
dialog=TextBlob("I hate movie 1. It was too violent ")

myview=TextBlob("I like autumn. It reminds me of some sad music ")
print(myview,":","\n",myview.sentiment,"\n")
dialog=dialog+myview

myview=TextBlob("The love story was cool too. A bit mushy but cool ")
print(myview,":","\n",myview.sentiment,"\n")
dialog=dialog+myview

myview=TextBlob("I would like to get out of here and see other horizons ")
print(myview,":","\n",myview.sentiment,"\n")
dialog=dialog+myview
     
#Parse noun phrases
print("Parse noun phrases to find potential key words:") 
print(dialog.noun_phrases)
  

m=input("Press ENTER if you agree to complete X's profiling dataset with some images")
print("The AI program will now enter social networks again and pick up KEY images")
print("that X commented with TAGS that fit the KEYWORDS found","\n")
print("CRL-MM Representation Learning Meta Model(see next section in book)")
print("The following image is a sample of the dataset of X. ")


take_images=input("Press ENTER to Continue")


directory='data/'
imgc=directory+'negative/lost.jpeg'
imgcc = load_img(imgc, target_size=(128, 128))
plt.imshow(imgcc)
plt.show(block=False)
time.sleep(5)
plt.close()

print("Now the Streaming website Chatbot will detect that a viewer")
print("has been scrolling for over 5 minutes and making no choice","\n")


mytext=input("Press ENTER to continue if you authorize the Chabot ask a profiled question...")


print("Hi, I am your personal assistant to find a movie you like")
print("I found a movie named Lost. The hero has to fight(action) through life")
print("after losing a loved one (love) and is searching for happiness(happiness) again...")
directory='data/'
imgc=directory+'negative/lost.jpeg'
imgcc = load_img(imgc, target_size=(128, 128))
plt.imshow(imgcc)
plt.show(block=False)
time.sleep(5)
plt.close()

mytext=input("Do you want to watch the movie?  ")
#enter I sure do!
myview=TextBlob(mytext)
print(myview,":",myview.sentiment)







