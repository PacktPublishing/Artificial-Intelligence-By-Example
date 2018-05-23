#Copyright 2018 Denis Rothman MIT License. See LICENSE.
from keras.models import model_from_json
from keras.models import load_model
import json
from pprint import pprint

#Directory
directory='dataset/' 
print("directory",directory)


#____________________LOAD MODEL___________________________
json_file = open(directory+'model/model.json', 'r')
loaded_jsonf = json_file.read()
loaded_json=json.loads(loaded_jsonf)
json_file.close()
print("MODEL:")
pprint(loaded_json)

#___________________ LOAD WEIGHTS_________________________

loaded_model=model_from_json(loaded_jsonf)
loaded_model.load_weights(directory+"model/model.h5")
print("WEIGHTS")
for layer in loaded_model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print(weights)
