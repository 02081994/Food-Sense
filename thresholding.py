from skimage.feature import local_binary_pattern
import scipy.stats
import numpy as np
import cv2

def process(original,th,filename):
	res = cv2.bitwise_and(original,original,mask = th)
	cv2.imshow("res",res)
	gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	# cv2.imshow("grayscale",gray)
	lbp_image=local_binary_pattern(gray,16,2,method='uniform')
	histogram=scipy.stats.itemfreq(lbp_image.ravel())
	mssg = "1"
	for i in range(0,len(histogram)):
		mssg = mssg+" "+str(i+1)+":"+str(histogram[i][1])
	mssg=mssg+"\n"
	filename.write(mssg)
	cv2.waitKey(0)

f1 = open("blotch_th1","a")
f2 = open("blotch_th2","a")
f3 = open("blotch_th3","a")
f4 = open("blotch_th4","a")

for i in range (3,81):

	print "1_data set/rot/rot ("+str(i)+").JPG"
	if(i<=68): original = cv2.imread("1_data set/rot/rot ("+str(i)+").JPG")
	else: original = cv2.imread("1_data set/rot/rot ("+str(i)+").jpg")
	img = cv2.cvtColor(original,cv2.COLOR_BGR2Lab)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow("original",original)
	cv2.imshow("gray",gray)
	cv2.imshow("res",img)
	cv2.waitKey(0)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# cv2.imshow("grayscale", gray)
	ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	ret,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	kernel = np.ones((5,5),np.uint8)

	th1 = cv2.bitwise_not(th1)
	process(original,th1,f1)


	th2 = cv2.bitwise_not(th2)
	th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
	process(original,th2,f2)
	# th2 = cv2.blur(th2,(5,5))
	

	th3 = cv2.bitwise_not(th3)
	th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
	process(original,th3,f3)
	# th3 = cv2.blur(th3,(5,5))
	

	th4 = cv2.bitwise_not(th4)
	process(original,th4,f4)


	# cv2.imshow("threshold1", res1)
	# res2 = cv2.bitwise_and(original,original,mask = th2)
	# cv2.imshow("threshold2", res2)
	# res3 = cv2.bitwise_and(original,original,mask = th3)
	# cv2.imshow("threshold3", res3)
	# res4 = cv2.bitwise_and(original,original,mask = th4)
	# cv2.imshow("threshold4", res4)

	# print histogram
f1.close()
f2.close()
f3.close()
f4.close()
