#Tensorboard reader
#Build with Tensorflow
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
def launchTensorBoard():
    import os
    PATH = os.getcwd()
    LOG_DIR = PATH+ '/LOGS/'
    os.system('tensorboard --logdir=' + LOG_DIR)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#and then click on Graph tab. You will then see the full graph.
