
#!/usr/bin/env python

'''
This sample demonstrates SEEDS Superpixels segmentation
Use [space] to toggle output mode
Usage:
  seeds.py [<video source>]
'''
from skimage.feature import local_binary_pattern
import scipy.stats
import numpy as np
import cv2

# relative module
#import video

# built-in module
import sys


if __name__ == '__main__':
    print __doc__

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('SEEDS')
    # cv2.createTrackbar('Number of Superpixels', 'SEEDS', 400, 1000, nothing)
    # cv2.createTrackbar('Iterations', 'SEEDS', 4, 12, nothing)

    seeds = None
    display_mode = 0
    num_superpixels = 400
    prior = 2
    num_levels = 4
    num_histogram_bins = 5

    # cap = cv2.VideoCapture(fn)
    img = cv2.imread("1_data set/blotch/blotch (20).JPG")

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    height,width,channels = converted_img.shape
    num_superpixels_new = 40 #cv2.getTrackbarPos('Number of Superpixels', 'SEEDS')
    num_iterations = 4 #cv2.getTrackbarPos('Iterations', 'SEEDS')

    if not seeds or num_superpixels_new != num_superpixels:
        num_superpixels = num_superpixels_new
        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
                num_superpixels, num_levels, prior, num_histogram_bins)
        color_img = np.zeros((height,width,3), np.uint8)
        color_img[:] = (0, 0, 255)

    seeds.iterate(converted_img, num_iterations)

    # retrieve the segmentation result
    labels = seeds.getLabels()
    print len(labels[1])

    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)


    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)
    # cv2.imshow("Original",original)
    if display_mode == 0:
        cv2.imshow('SEEDS', result)
    elif display_mode == 1:
        cv2.imshow('SEEDS', mask)
    else:
        cv2.imshow('SEEDS', labels)

    cv2.waitKey(0)
    # if ch == 27:
    #     break
    # elif ch & 0xff == ord(' '):
    #     display_mode = (display_mode + 1) % 3

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grayscale", gray)
    # ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # ret,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # kernel = np.ones((5,5),np.uint8)

    # th1 = cv2.bitwise_not(th1)
    # th2 = cv2.bitwise_not(th2)
    # th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    # # th2 = cv2.blur(th2,(5,5))
    # th3 = cv2.bitwise_not(th3)
    # th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    # # th3 = cv2.blur(th3,(5,5))
    # th4 = cv2.bitwise_not(th4)

    # res = cv2.bitwise_and(original,original,mask = th1)
    # cv2.imshow("threshold1", res)
    # res = cv2.bitwise_and(original,original,mask = th2)
    # cv2.imshow("threshold2", res)
    # res = cv2.bitwise_and(original,original,mask = th3)
    # cv2.imshow("threshold3", res)
    # res = cv2.bitwise_and(original,original,mask = th4)
    # cv2.imshow("threshold4", res)

    lbp_image=local_binary_pattern(gray,8,2,method='nri_uniform')
    histogram=scipy.stats.itemfreq(lbp_image.ravel())
    print len(histogram)
    # cv2.imshow("lbp", lbp_image)

