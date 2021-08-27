import cv2
import numpy as np

def imnormalize(img,mean,std,to_rgb=True):
    img = img.copy().astype(np.float32)
    return imnormalize_(img,mean,std,to_rgb)

def imnormalize_(img,mean,std,to_rgb):
    mean = np.float64(mean.reshape(1,-1))
    stdinv = 1/np.float64(std.reshape(1,-1))
    if to_rgb:
        cv2.cvtColor(img,cv2.COLOLR_BGR2RGB,img)#implace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img