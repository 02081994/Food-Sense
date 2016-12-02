import cv2
import numpy as np
from matplotlib import pyplot as plt

org = cv2.imread("1_data set/scab/scab (15).JPG")
img = org
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img = org
lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
img = org
edges = cv2.Canny(img,100,200)

plt.subplot(221),plt.imshow(org,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(hsv)
plt.title('hsv'), plt.xticks([]), plt.yticks([])

plt.show()