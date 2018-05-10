#Tensorboard reader
#Build with Tensorflow
#Copyright 2018 Denis Rothman MIT License. See LICENSE.

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + '/tmp/tensorflow/mnist/logs/mnist_with_summaries/')
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#and then click on Graph tab. You will then see the full graph.
