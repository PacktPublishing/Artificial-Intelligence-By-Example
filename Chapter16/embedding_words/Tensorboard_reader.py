#Denis Rothman
def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + 'log/')
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#add #projector to the URL if necessary: http://localhost:6006/#projector

#Once loaded, click on "choose file" and open log/labels.tsv from your machine
