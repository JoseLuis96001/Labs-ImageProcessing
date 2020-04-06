# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from math import sqrt, e

def shiftDFT(fImage):
    print(fImage.shape)
    print("-----------------------------------------------------------")
    fImage = fImage[0:fImage.shape[0] & -2, 0:fImage.shape[1] & -2]
    print(fImage.shape)
    cx = fImage.shape[1] // 2
    cy = fImage.shape[0] // 2

    q0 = fImage[0:cy, 0:cx]
    q1 = fImage[0:cy, cx:cx+cx]
    q2 = fImage[cy:cy+cy, 0:cx]
    q3 = fImage[cy:cy+cy, cx:cx+cx]
    
    tmp = np.zeros(q0.shape, dtype=q0.dtype)
    np.copyto(tmp,q0)
    np.copyto(q0,q3)
    np.copyto(q3,tmp)
    
    np.copyto(tmp,q1)
    np.copyto(q1,q2)
    np.copyto(q2,tmp)
    return fImage

def create_spectrum_magnitude_display(complexImg, rearrange):
    (planes_0, planes_1) = cv.split(complexImg)
    planes_0 = cv.magnitude(planes_0, planes_1)
   
    mag = planes_0.copy()
    mag += 1
    mag = cv.log(mag)

    if (rearrange):
        shiftDFT(mag);

    mag = cv.normalize(mag,  mag, 0, 1, cv.NORM_MINMAX)
    return mag

def create_ButterworthLowpassFilter(dft_Filter, D, n, W):
    tmp = np.zeros((dft_Filter.shape[0] & -2,dft_Filter.shape[1] & -2), dtype='float32')

    centre = ((dft_Filter.shape[0] & -2) // 2, (dft_Filter.shape[1] & -2) // 2)
    
    for i in range(dft_Filter.shape[0] & -2):
        for j in range(dft_Filter.shape[1] & -2):
            radius = sqrt(pow((i - centre[0]), 2) + pow((j - centre[1]), 2))
            try:
                tmp[i,j] = 1 / (1 + pow((radius /  D), (2 * n)))
            except:
                tmp[i,j] = 0 

    dft_Filter = cv.merge((tmp,tmp))
    return dft_Filter


   
image = cv.imread("lena.jpg" , 0)

filterOutput = np.array([])
padded = np.zeros(image.shape, dtype=image.dtype)



radius = 20           
order = 1              
width = 3

originalName = "Original image"
spectrumMagName = "Magnitude Image (log transformed)- spectrum"
lowPassName = "Butterworth Low Pass Filtered (grayscale)"
filterName = "Filter Image"
 

  
cv.namedWindow(originalName, cv.WINDOW_NORMAL)
cv.resizeWindow(originalName, 450,450)

cv.namedWindow(spectrumMagName, cv.WINDOW_NORMAL)
cv.resizeWindow(spectrumMagName, 450,450)

cv.namedWindow(lowPassName, cv.WINDOW_NORMAL)
cv.resizeWindow(lowPassName, 450,450)

cv.namedWindow(filterName, cv.WINDOW_NORMAL)
cv.resizeWindow(filterName, 450,450)


cv.imshow(originalName, image)

M = cv.getOptimalDFTSize(image.shape[0])
N = cv.getOptimalDFTSize(image.shape[1])   

padded = cv.copyMakeBorder(image, 0, M - image.shape[0], 0, N - image.shape[1], cv.BORDER_CONSTANT, value=0)
planes_0 = np.array(padded, dtype='float32')
planes_1 = np.zeros(padded.shape, dtype='float32')
complexImg = cv.merge((planes_0,planes_1))
complexImg = cv.dft(complexImg)

filter = complexImg.copy()
filter = create_ButterworthLowpassFilter(filter, radius, order, width)

complexImg = shiftDFT(complexImg)
complexImg = cv.mulSpectrums(complexImg, filter, 0)
complexImg = shiftDFT(complexImg)

mag = create_spectrum_magnitude_display(complexImg, True)

result = cv.idft(complexImg)

(myplanes_0,myplanes_1) = cv.split(result)
result = cv.magnitude(myplanes_0,myplanes_1)
result = cv.normalize(result,  result, 0, 1, cv.NORM_MINMAX)
imageRes = result            

(planes_0,planes_1) = cv.split(filter)
filterOutput = cv.normalize(planes_0, filterOutput, 0, 1, cv.NORM_MINMAX)

cv.imshow(spectrumMagName, mag)
cv.imshow(lowPassName, imageRes)
cv.imshow(filterName, filterOutput)



cv.waitKey(0)
cv.destroyAllWindows()



