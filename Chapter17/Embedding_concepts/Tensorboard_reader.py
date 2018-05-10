def launchTensorBoard():
    import os
    PATH = os.getcwd()   
    os.system('tensorboard --logdir=' + 'embedding-logs')

    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

#In your browser, enter http://localhost:6006 as the URL
#and then click on Graph tab. You will then see the full graph.
