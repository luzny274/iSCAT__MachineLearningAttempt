import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import shift
from PIL import Image
import os

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
    
#%%
#Load PSF

f = open("params.txt", "r")
focus = int(f.readline())
maxval = float(f.readline())
minval = float(f.readline())
f.close()

print("***Loading PSF***")

PSF = np.load("PSF.npy", allow_pickle=False)
PSF = PSF.astype('float64')
PSF = (PSF / PSF.max()) * (maxval - minval) + minval
PSF = PSF / (PSF.max() - PSF.min())
print("max=" + str(PSF.max()), "min=" + str(PSF.min()), "mean=" + str(PSF.mean()))

r = 10
showSlices(PSF, focus, "PSF")

mask = np.zeros(PSF.shape)

for x in range(mask.shape[1]):
    for y in range(mask.shape[2]):
        nx = x - mask.shape[1] / 2
        ny = y - mask.shape[2] / 2
        vzd = np.sqrt((nx**2 + ny**2)) - (mask.shape[1] / 2 - r)
        
        if vzd >= r:
            mask[0, x, y] = 0
        elif vzd > 0:
            mask[0, x, y] = -vzd * (1.0 / r) + 1
        else:
            mask[0, x, y] = 1

for z in range(mask.shape[0]):
    mask[z, :, :] = mask[0, :, :]
showSlices(mask, focus, "mask")

PSF = PSF * mask

showSlices(PSF, focus, "PSF")
print(PSF.shape)


print("***PSF loaded***")
#%%
#Load MTs
print("***Loading Microtubule PSF***")
MTs = list()

for file in os.listdir("MTs"):
    if file.endswith(".npy"):
        print("Loading " + file)
        MT = np.load("MTs/" + file, allow_pickle=False)
        MT = MT.astype('float64')
        MT = MT / (MT.max() - MT.min()) 
        MT = MT - MT[0,0,0]
        showSlices(MT, focus, file)
        MTs.append(MT)
        print("max=" + str(MT.max()), "min=" + str(MT.min()), "mean=" + str(MT.mean()))

print("***MTs loaded***")

#%%
#Generate Guassian mask

gauss_res = 512

def gaussPDF(r, std):
    ex = 0
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (r - ex)**2 / (2 * std**2))
    return y_out

def genGaussMask(res):
    mask = np.zeros((res, res))
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            nx = x - mask.shape[0] / 2
            ny = y - mask.shape[1] / 2
            vzd = np.sqrt((nx**2 + ny**2))
            mask[x, y] = gaussPDF(vzd, gauss_res / 2)
    
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

global_gaussian_mask = genGaussMask(gauss_res)
showArray(global_gaussian_mask, "gaussian mask")

def getRandomGaussMask(res):
    sz = np.random.randint(3 * gauss_res / 8, gauss_res / 2 + 3 * gauss_res / 8)
    
    var = gauss_res / 16
    offx1 = np.random.randint(-var, var) + (gauss_res - sz) // 2
    offy1 = np.random.randint(-var, var) + (gauss_res - sz) // 2
    
    result = global_gaussian_mask[offx1 : offx1+sz, offy1 : offy1+sz]
    return np.array(Image.fromarray(result).resize((res, res)))
    
#showArray(getRandomGaussMask(224), "random gauss mask")

#%%

from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)

import os
import re    

def create_path(path):
    isExist = os.path.exists(path)    
    if not isExist:      
        os.makedirs(path)
        
def sort_put(folder, img, name, vz):
    p_name = f'{vz:01d}'
    sorted_path = folder + "/" + p_name

    create_path(sorted_path)
    img.save(sorted_path + "/" + name)

def save_arr(arr, folder, name):
    I8 = arr.astype(np.uint8)
    img = Image.fromarray(I8)
    if (arr.max() - arr.min()) > 0:
        I8 = (((arr - arr.min()) / (arr.max() - arr.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(I8)
    img.save(folder + "/" + name)
    

from scipy import ndimage, misc

def crop_center(img, targetx, targety):
    newimg = np.zeros((targetx, targety))
    
    offx = img.shape[0]//2-(targetx//2)
    offy = img.shape[1]//2-(targety//2)   
    offnx = 0
    offny = 0
    
    if(offx < 0):
        offnx = -offx
        offx = 0
    if(offy < 0):
        offny = -offy
        offy = 0
        
    offx2  = min(img.shape[0], (offx + targetx))
    offy2  = min(img.shape[1], (offy + targety))
    offnx2 = offnx + (offx2 - offx)
    offny2 = offny + (offy2 - offy)
    newimg[offnx:offnx2, offny:offny2] = img[offx:offx2, offy:offy2]
    
    return(newimg)


def getNP(res, z):
    dev = res
    npx = int(np.random.uniform(-dev, dev) / 1.2)
    npy = int(np.random.uniform(-dev, dev) / 1.2)
    off = int((PSF.shape[1] - res) / 2)
    #nanoparticle = shift(PSF[z, off:(PSF.shape[1] - off), off:(PSF.shape[1] - off)], (npx,npy), cval=0)
    
    nanoparticle = crop_center(PSF[z, :, :], PSF.shape[1] + 2 * np.abs(npx), PSF.shape[2] + 2 * np.abs(npy))
    nanoparticle = ndimage.shift(nanoparticle, np.array([npx, npy]))
    nanoparticle = crop_center(nanoparticle, res, res)
    
    return [nanoparticle, (npx, npy)]


def getMT(res, z):
    dev = res
    mtx = int(np.random.uniform(-dev, dev) / 1.2)
    mty = int(np.random.uniform(-dev, dev) / 1.2)
    off = int((PSF.shape[1] - res) / 2)
    
    MTlen = np.random.randint(0, len(MTs))
    angle = np.random.uniform(0.0, 180.0)
    
    mt = ndimage.rotate(MTs[MTlen][z,:,:], angle, reshape=True)
    
    mt = crop_center(mt, mt.shape[0] + 2 * np.abs(mtx), mt.shape[1] + 2 * np.abs(mty))
    mt = ndimage.shift(mt, np.array([mtx, mty]))
    
    mt = crop_center(mt, res, res)

    #nanoparticle = shift(PSF[z, off:(PSF.shape[1] - off), off:(PSF.shape[1] - off)], (npx,npy), cval=0)
    
    return [mt, (mtx, mty, angle)]

import csv

ide = 0

def mynormal(EX, var, max_deviation):
    val = np.random.normal(EX, var)
    while(val > EX + max_deviation or val < EX - max_deviation):
        val = np.random.normal(EX, var)
    return val


import timeit
import time

def generateSamples(start_ide, speckles, writer, num_reg = 5000, foc_tol = 2, sortedFolder = "Sorted", samplesFolder = "Samples", max_garbages = 2, draw_gar_backg = False, draw_perlin_backg = False, saveDetails = False):
    
    start = time.time_ns()
    #Params
    target_res = 224
    
    max_focus = 60
    focus_scale = 1
    
    ex_contrast = 1.0
    dev_contrast = 0.9
    
    min_particles = 0
    max_particles = 2
    
    min_mts = 0
    max_mts = 2
    
    
    min_res = 196
    max_res = 512
    
    min_garbages = 0
    
    garbage_contrast_scale = 4.0
    
    garbage_background_contrast_min = 0.3
    garbage_background_contrast_max = 0.9
    
    perlin_background_contrast_min = 0.3
    perlin_background_contrast_max = 1.2
    ##
    
    
    ide = start_ide

    #Function
    sz = speckles.shape
    

    for _ in range(num_reg):
        
        res = np.random.randint(min_res, max_res + 1)
        minv = 10
        maxv = sz[1] - minv - res
    
        vx = int(np.random.uniform(minv, maxv))
        vy = int(np.random.uniform(minv, maxv))
        
        focused = np.random.randint(0, 2)
        vz = focus
        if focused != 0:
            vz = np.random.randint(-max_focus, max_focus - 2 * foc_tol) * focus_scale
            if vz >= -foc_tol:
                vz = vz + 2 * foc_tol + 1
            vz = vz + focus
        
        #Get speckle pattern
        background = speckles[vz, vx:(vx+res), vy:(vy+res)]
        background = background / background.max()
        
        foreground = np.zeros((res, res))
        garbage = np.zeros((res, res))
        
        garbage_background = np.zeros((res, res))
        perlin_background = np.zeros((res, res))
        
        #Get nanoparticles
        num_particles = np.random.randint(min_particles, max_particles + 2)
        for i in range(num_particles):
            contrast = mynormal(0, (dev_contrast) * 2.0, dev_contrast)
            contrast = round(contrast + ex_contrast, 3)
    
            nanoparticle = getNP(res, vz)[0]
            nanoparticle = nanoparticle * contrast
            foreground = foreground + nanoparticle
            
        #Get microtubules
        num_mts = np.random.randint(min_mts, max_mts + 2)
        for i in range(num_mts):
            contrast = mynormal(0, (dev_contrast) * 2.0, dev_contrast)
            contrast = round(contrast + ex_contrast, 3)
    
            mt = getMT(res, vz)[0]
            mt = mt * contrast
            foreground = foreground + mt
            
        #Get garbage
        num_garbages = np.random.randint(min_garbages - max_garbages / 2.0, max_garbages + 2)
        if(num_garbages < min_garbages):
            num_garbages = min_garbages
        for i in range(num_garbages):        
            contrast = mynormal(0, (dev_contrast) * 2.0 * garbage_contrast_scale, dev_contrast * garbage_contrast_scale)
            contrast = round(contrast + ex_contrast * garbage_contrast_scale, 3)
    
            gar_res = np.random.randint(res / 2.0, res * 2.0)
            z = np.random.randint(focus - 50, focus + 50)
            nanoparticle = getNP(gar_res, z)[0]
            nanoparticle = nanoparticle * contrast
            
            nanoparticle = np.array(Image.fromarray(nanoparticle).resize((res, res)))
            garbage = garbage + nanoparticle
            
        #Get garbage background
        if(draw_gar_backg):
            index = np.random.randint(0, len(specklesList))
            gar_res = np.random.randint(res / 2.0, res * 2.0)
            
            vx2 = int(np.random.uniform(minv, sz[1] - minv - gar_res))
            vy2 = int(np.random.uniform(minv, sz[1] - minv - gar_res))
            
            vz2 = np.random.randint(0, specklesList[index].shape[0])
            
            contrast = np.random.uniform(garbage_background_contrast_min, garbage_background_contrast_max)
            garbage_background = specklesList[index][vz2, vx2:(vx2+gar_res), vy2:(vy2+gar_res)]
            
            if((garbage_background.max() - garbage_background.min()) > 0):
                garbage_background = (garbage_background - garbage_background.min()) / (garbage_background.max() - garbage_background.min()) * contrast
            else:
                garbage_background = np.zeros((res, res))
            
        #Get perlin noise
        if(draw_perlin_backg):
            a = pow(2, np.random.randint(0, 4))
            b = np.random.randint(1, 6)
            contrast = np.random.uniform(perlin_background_contrast_min, perlin_background_contrast_max)
            perlin_background = generate_fractal_noise_2d((512, 512), (a, a), b)
            perlin_background = (perlin_background - perlin_background.min()) / (perlin_background.max() - perlin_background.min()) * contrast
            
        
        #Resize arrays to target resolution       
        background          = np.array(Image.fromarray(background).resize((target_res, target_res)))
        foreground          = np.array(Image.fromarray(foreground).resize((target_res, target_res)))
        garbage             = np.array(Image.fromarray(garbage).resize((target_res, target_res)))
        garbage_background  = np.array(Image.fromarray(garbage_background).resize((target_res, target_res)))
        perlin_background   = np.array(Image.fromarray(perlin_background).resize((target_res, target_res)))
        
        gauss_mask = getRandomGaussMask(target_res)
        
        result = background + foreground + garbage + garbage_background + perlin_background
        result = result * gauss_mask
        
        #Get noise
        NSexp = np.random.uniform(2.2, 4.0)
        NSrel = np.power(5.0, NSexp)

        result = (result - result.min()) / (result.max() - result.min())
        result = result + 0.0000001
        scale = result.max()
        
        noisy = np.random.poisson((result / scale) * NSrel) / NSrel * scale
        
        
        #Result
        ide = ide + 1
        #showArray(garbage, "garbage" + str(ide))
        #showArray(foreground, "foreground" + str(ide))
        #showArray(result, "result" + str(ide))
        #showArray(noisy, "noisy" + str(ide))
        
        I8 = (((noisy - noisy.min()) / (noisy.max() - noisy.min())) * 255.9).astype(np.uint8)
        img = Image.fromarray(I8)
        
        
        sgn = "+"
        if(vz - focus < 0):
            sgn = ""
        
        #Save result
        create_path(sortedFolder)
        
        file_name = str(ide) + "_main_focus=" + sgn + str(int((vz - focus) / focus_scale)) + ".png"
        sort_put(sortedFolder, img, file_name, focused)
        
        if(saveDetails):      
            create_path(samplesFolder)
            
            #Save arrays
            ##Main
            save_arr(noisy, samplesFolder, str(ide) + "_main_focus=" + sgn + str(int((vz - focus) / focus_scale)) + ".png")
            ##Details
            save_arr(foreground, samplesFolder, str(ide) + "_foreground" + ".png")
            save_arr(background, samplesFolder, str(ide) + "_background" + ".png")
            save_arr(result, samplesFolder, str(ide) + "_no_noise" + ".png")
            save_arr(garbage, samplesFolder, str(ide) + "_garbageNPs" + ".png")
            save_arr(garbage_background, samplesFolder, str(ide) + "_garbage_background" + ".png")
            save_arr(perlin_background, samplesFolder, str(ide) + "_perlin_background" + ".png")
            save_arr(gauss_mask, samplesFolder, str(ide) + "_gauss_mask" + ".png")
            
            #Save csv
            f = open(samplesFolder + "/" + str(ide) + "_params.txt", "w")
            f.write("focus = " + str(int((vz - focus) / focus_scale)) + '\n')
            f.write("num_particles = " + str(num_particles) + '\n')
            f.write("num_mts = " + str(num_mts) + '\n')
            f.write("num_garbages = " + str(num_garbages) + '\n')
            f.write("NSrel = " + str(NSrel) + '\n')
            f.close()
            
            writer.writerow([str(ide), sgn + str(int((vz - focus) / focus_scale))])
        
    
    end = time.time_ns()
    perf = end - start
    print("*** time: " + str(perf / 1000000000.0) + " s")
    return 1
      
        
#***************************************************
#%%

for idx, speckles in enumerate(specklesList):
    speckles = speckles.astype('float64')
    print("speckle pattern " + str(idx) + "/" + str(len(specklesList)))
    
    num_reg = 20000
    generateSamples(idx * num_reg, speckles, None, num_reg = num_reg, foc_tol = 2, sortedFolder = "Binary_Sorted", samplesFolder = "Binary_Samples", max_garbages = 3, draw_gar_backg = True, draw_perlin_backg = True,  saveDetails = False)
    
    
#***************************************************

#%%
#Load speckles first

specklesList2 = list()

import os
for file in os.listdir("Speckles"):
    if file.endswith(".npy"):
        print("Loading speckles " + str(file))
        speckles = np.load("Speckles/" + file, allow_pickle=False)
        print("Loaded")  
        specklesList2.append(speckles)
        
specklesList = specklesList2