import numpy as np
import cv2

original = cv2.imread('1_data set/blotch/blotch(16).JPG')

img = cv2.cvtColor(original,cv2.COLOR_BGR2Lab)

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
 
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
# print center
print len(label)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow("Original",original)
cv2.imshow("res2",res2)
cv2.imshow('res3',cv2.cvtColor(res2,cv2.COLOR_Lab2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()