
#first step of importing before constuction of CNN network


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


sigmalist=[8]
gammalist=[0.5]
blist=[1,1.8,2.6]
thetalist=[0,np.pi/4, 2*np.pi/4, 3*np.pi/4]
philist=[np.pi/2, 3*np.pi/2]
def calc_lambda(sigma, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)
tl=[str(0), chr(960)+"/4", "2"+chr(960)+"/4","3"+chr(960)+"/4"]
pl=[chr(960)+"/2","3"+chr(960)+"/2"]
picsave=[]

fig=plt.figure()

plt.subplots_adjust(0, 0, 2, 1)

od=0
op=0
i=0
for b in blist:
    j=0
    op=0
    for theta in thetalist:
        for phi in philist:
            plt.subplot2grid((3, 8), (i, j))
            Lambda=calc_lambda(sigmalist[0],b)
            kernel= cv2.getGaborKernel((31,31), sigmalist[0], theta, Lambda, gammalist[0], phi)
            plt.xticks([])
            plt.yticks([])
            if od==0:
                plt.title(chr(952)+"="+tl[j//2]+"  "+chr(968)+"="+pl[j%2])
            if op==0:
                plt.ylabel("b="+str(b))
                op+=1
            plt.imshow(kernel,cmap="rainbow")
            #plt.colorbar()
            j+=1
    i+=1
    od+=1
#fig.colorbar()
plt.show()


#Parameter sigma: The bandwith or sigma controls the overall size of the Gabor envelope. For larger bandwith the envelop increases allowing more stripes and with small bandwith the envelope tightens
#On increasing the sigma to 30 and 45

j=0
for i in sigmalist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(i,1.8)
    kernel= cv2.getGaborKernel((31,31), i, 0, Lambda, 0.5, np.pi/2)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("sigma="+str(i))
    j+=1


sigmalist=[8]
gammalist=[0.5]
blist=[1,1.8,2.6]
thetalist=[0,np.pi/4, 2*np.pi/4, 3*np.pi/4]
philist=[np.pi/2, 3*np.pi/2]

j=0
blist=[0.5,1.2,1.8,2.3,3]
for i in blist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(8,i)
    kernel= cv2.getGaborKernel((31,31), 8, 0, Lambda, 0.5, np.pi/2)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("b"+str(i))
    j+=1


# Parameter gamma: The aspect ratio or gamma controls the height of the Gabor function. For very high aspect ratio the height becomes very small and for very small gamma value the height becomes quite large. On increasing the value of Gamma to 0.5 and 0.75,
#Keeping others unchanged, the height of the Gabor function reduces

j=0
gammalist=[0.2,0.35,0.5,0.75,1]
for i in gammalist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(8,1.8)
    kernel= cv2.getGaborKernel((31,31), 8, 0, Lambda, i, np.pi/2)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("gamma="+str(i))
    j+=1

#Parameter Theta: Controls the orientation of the Gabor function with respect to 2-D space. the zero degree that corresponds to the vertical position of the Gabor function

j=0
thetalist=[0,1/5*np.pi,2/5*np.pi,3/5*np.pi,4/5*np.pi]
for i in thetalist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(8,1.8)
    kernel= cv2.getGaborKernel((31,31), 8, i, Lambda, 0.5, np.pi/2)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("theta="+str(round(i,3)))
    j+=1

#Parameter phi: the phase offset of the sinusoidal function

j=0
philist=[0,1/5*np.pi,2/5*np.pi,3/5*np.pi,4/5*np.pi]
for i in philist:
    plt.subplot2grid((1, 5), (0, j))
    Lambda=calc_lambda(8,1.8)
    kernel= cv2.getGaborKernel((31,31), 8, 0, Lambda, 0.5, i)
    plt.imshow(kernel,cmap="rainbow")
    plt.xlabel("phi="+str(round(i,3)))
    j+=1