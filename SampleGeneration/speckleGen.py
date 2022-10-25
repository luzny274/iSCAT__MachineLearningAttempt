import numpy as np
import matplotlib.pylab as plt
from scipy import signal
%matplotlib inline

#Head

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

showSlices(PSF, focus, "PSF")

#%%
#Surface Generation test

scale = 2
surfaceSZ = (PSF.shape[1] * scale, PSF.shape[2] * scale)

#povrchval = np.random.sample((sz[1], sz[2])) * 3
expectedValue = 1
variation = 1
povrchval = np.random.lognormal(expectedValue, variation, surfaceSZ)
povrchval[povrchval > 1000] = expectedValue

povrchval = povrchval / povrchval.max()
showArray(povrchval, "Surface roughness")

#%%
#Speckle Generation test

depth = int(PSF.shape[0] / 1)

sz = (depth, surfaceSZ[0], surfaceSZ[1])

povrch = np.zeros(sz)
povrch[int(sz[0] / 2), :, :] = povrchval

print("convolving...")
speckles = signal.oaconvolve(povrch, PSF, mode='same')
print("convolved")

showSlices(speckles, focus - int((PSF.shape[0] - depth) / 2), "Speckles")

#%%

#Generation of Speckles

def generateSpeckles(EX, Var, limit, w, h):
    limit = int(limit)
    w = int(w)
    h = int(h)
    surfaceSZ = (w, h)

    povrchval = np.random.lognormal(EX, Var, surfaceSZ)
    povrchval[povrchval > limit] = EX

    povrchval = povrchval / povrchval.max()
    showArray(povrchval, "Surface roughness")
    
    sz = (PSF.shape[0], surfaceSZ[0], surfaceSZ[1])
    
    povrch = np.zeros(sz)
    povrch[int(sz[0] / 2), :, :] = povrchval
    
    pri,nt("convolving...")
    speckles = signal.convolve(povrch, PSF, mode='same')
    print("convolved")
    
    
    speckles = (((speckles - speckles.min()) / (speckles.max() - speckles.min())) * 255.9).astype(np.uint8)
    np.save("Speckles/EX=" + str(EX) + "_var=" + str(Var) + "_limit=" + str(limit) + ".npy", speckles, allow_pickle=False)

    showSlices(speckles, focus, "Speckles")

for EX in np.linspace(0.5, 2.0, 3):
    for var in np.linspace(0.5, 3.0, 4):
        for limit in np.linspace(1000, 2000, 2):
            generateSpeckles(EX, var, limit, 2048, 2048)


#%%
#Generated Speckles Viewer

def viewSpeckle(path):
    print("Loading speckles...")
    speckles = np.load(path, allow_pickle=False)
    #showSlices(speckles, focus, path)
    print("Plotting speckles...\n")
    showArray(speckles[focus, 1:200, 1:200], path)


import os
for file in os.listdir("Speckles"):
    if file.endswith(".npy"):
        viewSpeckle("Speckles/" + file)

