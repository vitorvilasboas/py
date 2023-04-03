# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:38:31 2019

@author: Rafael Rocha
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
from glob import glob
from skimage.transform import resize 
from skimage.io import imread, imshow
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float
import cv2

img = imread('images/astronaut_gray.png')

plt.subplot(221)
imshow(img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

plt.subplot(222)
imshow(sobelx, cmap='gray')

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
plt.subplot(223)
imshow(sobely, cmap='gray')

canny = cv2.Canny(img, 100, 200)
plt.subplot(224)
imshow(canny, cmap='gray')