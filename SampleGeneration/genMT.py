import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import shift
from PIL import Image
import cv2
import os
from scipy import signal
%matplotlib inline

#HEAD


def showSlices(arr, focus, title):
    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    sz = arr.shape
    ax_1 = fig.add_subplot(131)
    ax_1.imshow(arr[:, int(sz[1] / 2), :])
    
    ax_2 = fig.add_subplot(132)
    ax_2.imshow(arr[:, :, int(sz[2] / 2)])
    
    ax_3 = fig.add_subplot(133)
    ax_3.imshow(arr[int(focus), :, :])
    
    # plt.savefig(filename)
    plt.title(title)
    plt.show()
    plt.close()

def showArray(arr, title):
    plt.imshow(arr, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()
    plt.close()

f = open("params.txt", "r")
focus = int(f.readline())
maxval = float(f.readline())
minval = float(f.readline())
f.close()

PSF = np.load("PSF.npy", allow_pickle=False)
PSF = PSF.astype('float64')
PSF = (PSF / PSF.max()) * (maxval - minval) + minval
PSF = PSF / (PSF.max() - PSF.min())

showSlices(PSF, focus, "PSF")

#%%
#generate microtubules


for length in np.linspace(64, 384, 24):
    length = int(length)
    print("len=" + str(length))
    sz = PSF.shape
    cyl = np.zeros((sz[0], sz[1] * 2, sz[2]))
    sz = cyl.shape
    
    startz = int(sz[0] / 2)
    starty = int(sz[2] / 2)
    
    off = int((sz[1] - length) / 2)
    for x in range(off, sz[1] - off):
        cyl[startz, x, starty] = 1
        
    #convolution
    
    cylconvoluted = signal.convolve(cyl, PSF, mode='same')
    
    r = 20
    for z in range(cylconvoluted.shape[0]):
        for x in range(cylconvoluted.shape[1]):
            for y in range(cylconvoluted.shape[2]):
                nx = min(x, cylconvoluted.shape[1] - x)
                ny = min(y, cylconvoluted.shape[2] - y)
                vzd =  r - min(nx, ny)
                
                if vzd > r:
                    cylconvoluted[z, x, y] = 0
                elif vzd > 0:
                    cylconvoluted[z, x, y] = cylconvoluted[z, x, y] * (-vzd * (1.0 / r) + 1)

    
    maxval = cylconvoluted.max()
    minval = cylconvoluted.min()   
    cylconvoluted = (((cylconvoluted - minval) / (maxval - minval)) * 255.9).astype(np.uint8)
    
    #saving
    np.save("MTs/len=" + str(length) +".npy", cylconvoluted, allow_pickle=False)
    f.close()
    
    showSlices(cylconvoluted, focus, "MT")
            