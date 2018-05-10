def launchTensorBoard():
    import os
    #os.system('tensorboard --logdir=' + 'c:/tfoutput/Test')
    os.system('tensorboard --logdir=' + 'c:/tmp/tensorflow/mnist/logs/mnist_with_summaries/')
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#and then click on Graph tab. You will then see the full graph.
