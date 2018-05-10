#Edge dection kernel
#Built with SciPy
#Copyright 2018 Denis Rothman MIT License. READ LICENSE.
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage.filters as filter
import matplotlib.pyplot as plt


#I.An edge dectection kernel
kernel_edge_detection = np.array([[0.,1.,0.],
                                [1.,-4.,1.],
                                [0.,1.,0.]])

#II.Load image
image=mpimg.imread('img.bmp')[:,:,0]
shape = image.shape
print("image shape",shape)
#III.Convolution
image_after_kernel = filter.convolve(image,kernel_edge_detection,mode='constant', cval=0)



#III.Displaying the image before and after the convolution
f = plt.figure(figsize=(8, 8))
axarr=f.subplots(2,sharex=False)
axarr[0].imshow(image,cmap=plt.cm.gray)
axarr[1].imshow(image_after_kernel,cmap=plt.cm.gray)
f.show()



print("image before convolution")
print(image)
print("image after convolution")
print(image_after_kernel)

