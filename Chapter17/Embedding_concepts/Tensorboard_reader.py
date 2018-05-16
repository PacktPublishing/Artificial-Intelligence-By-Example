def launchTensorBoard():
    import os
    PATH = os.getcwd()   
    os.system('tensorboard --logdir=' + 'embedding-logs')

    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#add #projector to the URL if necessary: http://localhost:6006/#projector
