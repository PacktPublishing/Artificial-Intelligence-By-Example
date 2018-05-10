def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + 'rlog/RNN/')
    #os.system('tensorboard --logdir=' + '/tmp/mnist_tutorial/')
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#and then click on Graph tab. You will then see the full graph.
