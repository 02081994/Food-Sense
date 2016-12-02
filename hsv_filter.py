import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result',cv2.WINDOW_NORMAL)

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)


frame = cv2.imread('abd.png')
avg = [0.0,0.0,0.0]
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([frame],[i],None,[256],[0,256])
    arr= np.array(histr)
    # print "see"
    avg[i] = 0.0
    totalNum = 0.0
    for j in range(0, len(arr)):
        avg[i] += j*arr[j]                  
        totalNum += arr[j]
    avg[i] /= totalNum
    # print arr
    # avg[i]=np.average(arr);
    # plt.plot(histr,color = col)
    # plt.xlim([0,256])
# plt.show()

print avg[0], avg[1], avg[2]
#converting to HSV
hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

# get info from track bar and appy to result
h = cv2.getTrackbarPos('h','result')
s = cv2.getTrackbarPos('s','result')
v = cv2.getTrackbarPos('v','result')



# Normal masking algorithm
lower_blue = np.array([h,s,v])
upper_blue = np.array([180,255,255])

mask = cv2.inRange(hsv,lower_blue, upper_blue)

result = cv2.bitwise_and(frame,frame,mask = mask)

cv2.imshow('result',result)

a = raw_input()

cap.release()

cv2.destroyAllWindows()