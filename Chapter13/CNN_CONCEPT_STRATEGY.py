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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import scipy.misc      
from PIL import Image
import time

directory='dataset/'

display=1     #display images     
print("directory",directory)

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

def identify(target_image):
    filename = target_image
    original = load_img(filename, target_size=(64, 64))
    #print('PIL image size',original.size)
    if(display==1):
        plt.imshow(original)
        plt.show(block=False)
        time.sleep(5)
        plt.close()
    numpy_image = img_to_array(original)
    arrayresized = scipy.misc.imresize(numpy_image, (64,64))    
    #print('Resized',arrayresized)
    inputarray = arrayresized[np.newaxis,...] # extra dimension to fit model 
    #print('newaxis',inputarray)
   
#___________________PREDICTION___________________________
    prediction1 = loaded_model.predict_proba(inputarray)
    prediction2 = loaded_model.predict(inputarray)
    print("image",target_image,"predict_proba:",prediction1,"predict:",prediction2)
    return prediction1


#___________________SEARCH STRATEGY_____________________


MS1='full'
MS2='available'

I=['1','2','3','4','5','6']

for im in range(2):
    imgc=directory+'classify/img'+ I[im] + '.jpg'
    print(imgc)
    s=identify(imgc)
    if (int(s)==0):
        print('Classified in class A')
        print(MS1)
    if (int(s)==1):
        print('Classified in class B')
        print(MS2)
