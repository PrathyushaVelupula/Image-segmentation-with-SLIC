
import argparse

import cv2

import numpy as np
from skimage.segmentation import slic


def alg_SLIC(image,r,size):
    img2 = 	cv2.ximgproc.createSuperpixelSLIC(image, region_size = size, ruler = r)
    img2 = slic(image, n_segments=100, compactness=10)
    img2.iterate(10)
    mask_img2 = img2.getLabelContourMask()
    segments=img2.getNumberOfSuperpixels()
    mask_inv_img2 = cv2.bitwise_not(mask_img2)
    color_img = cv2.cvtColor(img2,cv2.COLOR_Lab2BGR)
    img_slic = cv2.bitwise_and(color_img,color_img,mask = mask_inv_img2)
    cv2.imwrite('SLIC.jpg', img_slic)
    cv2.imshow("img_slic",img_slic)
    cv2.waitKey(0)
    cv2.imwrite('SLIC.jpg', img_slic)


def alg_SLICO(image,r,size):
    img2 = 	cv2.ximgproc.createSuperpixelSLIC(image, algorithm = cv2.ximgproc.SLICO, region_size = size, ruler = r)
    img2.iterate(10)
    mask_img2 = img2.getLabelContourMask()
    mask_inv_img2 = cv2.bitwise_not(mask_img2)
    color_img = cv2.cvtColor(img2,cv2.COLOR_Lab2BGR)
    img_slico = cv2.bitwise_and(image,image,mask = mask_inv_img2)
    cv2.imwrite('SLICO.jpg', img_slico)
    cv2.imshow("img_slico",img_slico)
    cv2.waitKey(0)
    

def alg_MSLIC(image,r,size):
    img2 = 	cv2.ximgproc.createSuperpixelSLIC(image, algorithm = cv2.ximgproc.MSLIC, region_size = size, ruler = r)
    img2.iterate(10)
    mask_img2 = img2.getLabelContourMask()
    mask_inv_img2 = cv2.bitwise_not(mask_img2)
    img_mslic = cv2.bitwise_and(image,image,mask = mask_inv_img2)
    cv2.imwrite('MSLIC.jpg', img_mslic)
    cv2.imshow("img_mslic",img_mslic)
    cv2.waitKey(0)

def main():
    my_parser = argparse.ArgumentParser(description = 'Simple Linear Iterative Clustering')
    my_parser.add_argument('-input',help = 'filename of image')
    my_parser.add_argument('--alg', help = 'Name of SLIC algorithm variant', required= False, type = str)
    my_parser.add_argument('--ruler', help = 'enter the average superpixel size measured in pixels', required=False, default=10 , type=int)
    my_parser.add_argument('--size', help = 'enter the enforcement of superpixel smoothness', required=False, default=10 , type=int)
    args = my_parser.parse_args()
    image = cv2.imread(args.input)
    cv2.imshow('IMAGE',image)
    cv2.waitKey()
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
    size = args.size 
    ruler = args.ruler
    alg = args.alg
    if alg == 'SLIC':
        alg_SLIC(image2,ruler,size)
    elif alg == 'SLICO':
        alg_SLICO(image,ruler,size)
    elif alg == 'MSLIC':
        alg_MSLIC(image,ruler,size)
    else :
        alg_SLIC(image,ruler,size)
    cv2.destroyAllWindows

if __name__ == "__main__":
    main()