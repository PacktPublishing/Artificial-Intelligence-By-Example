#Copyright 2018 Denis Rothman MIT License. See LICENSE.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import save_model
from keras.models import load_model
from keras import backend as K
from pprint import pprint


A=['dataset_O/','dataset_traffic/','dataset/']     
scenario=2            #reference to A
directory=A[scenario] #transfer learning parameter (choice of images)
print("directory",directory)

# Part 1 - Building the CNN

estep=100 #8000
batchs=10 #32
vs=100    #2000
ep=2      #25

model_file_name = 'classifier.h5'
#https://github.com/keras-team/keras/issues/5073

# Initializing the CNN
print("Step 0 Initializing")
classifier = Sequential()

# Step 1 - Convolution
print("Step 1 Convolution")
classifier.add(Conv2D(32, (3, 3),input_shape = (64, 64, 3), activation = 'relu'))
 
# Step 2 - Pooling
print("Step 2 MaxPooling2D")
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
print("Step 3a Convolution")
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
imp=classifier.input
outputs = [layer.output for layer in classifier.layers] 

print("Step 3b Pooling")
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Flattening
print("Step 4 Flattening")
classifier.add(Flatten())

# Step 4 - Full connection
print("Step 5 Dense")
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
print("Layer information Summary")
imp=classifier.input
outputs = [layer.output for layer in classifier.layers]
print("input features:")
pprint(imp)
print("output features")
pprint(outputs)

# Compiling the CNN
print("Step 6 Optimizer")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

print("Step 7a train")
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

print("Step 7b training set")
training_set = train_datagen.flow_from_directory(directory+'training_set',
                                                 target_size = (64, 64),
                                                 batch_size = batchs,
                                                 class_mode = 'binary')

print("Step 8a test")
test_datagen = ImageDataGenerator(rescale = 1./255)


print("Step 8b testing set")
test_set = test_datagen.flow_from_directory(directory+'test_set',
                                            target_size = (64, 64),
                                            batch_size = batchs,
                                            class_mode = 'binary')
print("Step 9 training")
print("Classifier",classifier.fit_generator(training_set,
                         steps_per_epoch = estep,
                         epochs = ep,
                         validation_data = test_set,
                         validation_steps = vs,verbose=2))


# serialize model to JSON
model_json = classifier.to_json()
with open(directory+"model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights(directory+"model/model.h5")
from keras.utils import plot_model
plot_model(classifier, to_file=directory+'model/model.png',show_shapes=True )
print("Model saved to disk")
