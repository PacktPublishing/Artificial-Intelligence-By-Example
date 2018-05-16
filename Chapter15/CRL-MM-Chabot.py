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
from textblob import TextBlob

directory='dataset/'
display=1     #display images     
#print("directory",directory)

#____________________LOAD MODEL____________________________
json_file = open(directory+'model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
#___________________ load weights into new model
loaded_model.load_weights(directory+"model/model.h5")
#print("Strategy model loaded from training repository.")
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
#    print("image",target_image,"predict_proba:",prediction1,"predict:",prediction2)
    return prediction1


#___________________SEARCH STRATEGY_____________________


MS1='This is sports coach.'
MS2='This is a bus.'

I=['1','2','3','4','5','6']

print("COGNITIVE NATURAL LANGUAGE PROCESSING")
print("Conceptual Representation Learning Meta Model(CRL-MM):","\n","concepts in words+images.","\n")
print("In chapter 8, the Google_Translate_Customized.py ","\n","program found an error in Google Translate.")
print("Google Translate confused a sports coach with a vehicle-coach.","\n","In the near future, Google will probably correct this example ","\n","but here are thousands of other confusions.")
print("On line 48 of Google_Translate_Customized.py, a deeper_translate function ","\n","was added to correct such errors. ","\n","This was only possible because Google provides ","\n","the necessary tools for the AI community to help make progress.")
print("The translation was corrected so that the word bus replaced coach","\n"," and Google translated the sentence correctly")
print("This program extends polysemy deeper understanding to concepts:","\n"," words and images","\n")
print("This model can be added to","\n","deep_translate in Google_Translate_Customized at line 114 or")
print("deployed in a cognitive chabot as follows...")
press=input("Press ENTER to continue")

print("\n","PERSON IN A BUS THAT BROKE DOWN TEXTING THE POLICE","\n","A MESSAGE WITH A CHABOT TRANSLATOR:","\n", "My sports coach broke down(the sentence is in French but came out that way)","\n","\n")
print("POLICE RECEIVING THE SMARTPHONE MESSAGE:")
text = "I dont understand what you mean."
obj = TextBlob(text)
sentiment = obj.sentiment.polarity
print (text,"\n"," The sentiment analysis estimation is: ",sentiment)
if(sentiment<0):
    print("IMAGE DISPLAYED: FROWN OR OTHER POLYSEMY")
    imgc=directory+'cchat/notok.jpg'
    imgcc = load_img(imgc, target_size=(64, 64))
    plt.imshow(imgcc)
    plt.show(block=False)
    time.sleep(5)
    plt.close()
yesno=0

for im in range(2):
    imgc=directory+'classify/img'+ I[im] + '.jpg'
    #print(imgc)
    s=identify(imgc)
    if (int(s)==0):
        print(MS1)
        print("Polysemy Activated","\n","POLICE: Do you mean this coach?")
        answer = input('Enter yes or no(show an icon also to avoid language): ')
    if(answer=='no'):
        if (int(s)==1):
            print(MS2)
            print("Polysemy activated:","\n","POLICE: Do you mean this coach?")
            answer = input('Enter yes or no: ')
            if(yesno=='no'):
                print("Add a definition of this concept")
    if(answer=='yes'):
        print("CHABOT:Then the best word would be :bus")
        print("\n","POLICE:I understand. Your bus broke down.")
        print("Bus+ brock down")
        imgc=directory+'cchat/img5.jpg'
        imgcc = load_img(imgc, target_size=(128, 200))
        plt.imshow(imgcc)
        plt.show(block=False)
        time.sleep(5)
        plt.close()
        print("Is this what you mean?")
        answer = input('Enter yes or no: ')
        if(answer=="yes"):
            print("THE SYSTEM WRITES A SENTENCE:","\n")
            text = "Image to text: Yes that is correct"
            obj = TextBlob(text)
            sentiment = obj.sentiment.polarity
            print (text," : ",sentiment)
            if(sentiment>=0):
                imgc=directory+'cchat/ok.jpg'
                imgcc = load_img(imgc, target_size=(64,64))
                plt.imshow(imgcc)
                plt.show(block=False)
                time.sleep(5)
                plt.close()
                print("\n","POLICE : OK. We are already on our way.","\n","The bus emergency sensor sent us a message 5 minutes ago.","\n","Thanks for confirming","\n")
                print("\n","POLICE : We will be there at 15:30")
                imgc=directory+'cchat/ETA.jpg'
                imgcc = load_img(imgc, target_size=(128, 200))
                plt.imshow(imgcc)
                plt.show(block=False)
                time.sleep(5)
                plt.close()
        if(answer=='no'):
            print("Add more dialogs to the program")
        break

    


