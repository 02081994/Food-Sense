import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100
hm,sm,vm = 180,255,255

# Creating track bar
# cv2.createTrackbar('h', 'result',0,179,nothing)
# cv2.createTrackbar('s', 'result',0,255,nothing)
# cv2.createTrackbar('v', 'result',0,255,nothing)
# cv2.createTrackbar('hm', 'result',0,179,nothing)
# cv2.createTrackbar('sm', 'result',0,255,nothing)
# cv2.createTrackbar('vm', 'result',0,255,nothing)


frame = cv2.imread("1_data set/normal/normal (26).JPG")
#converting to HSV
hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

# get info from track bar and appy to result
# h = cv2.getTrackbarPos('h','result')
# s = cv2.getTrackbarPos('s','result')
# v = cv2.getTrackbarPos('v','result')
# hm = cv2.getTrackbarPos('hm','result')
# sm = cv2.getTrackbarPos('sm','result')
# vm = cv2.getTrackbarPos('vm','result')

# Normal masking algorithm
lower_blue = np.array([0,0,0])
upper_blue = np.array([10,141,174])

mask = cv2.inRange(hsv,lower_blue, upper_blue)

result = cv2.bitwise_and(frame,frame,mask = mask)

cv2.imshow('result',result)

cv2.waitKey(0)
# if k == 27:
#     break

# cap.release()

# cv2.destroyAllWindows()