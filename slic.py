# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
from skimage.util import img_as_float
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2



f1 = open("data.txt","a",0)
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())

# load the image and apply SLIC and extract (approximately)
# the supplied number of segments

for i in range (74,81):

	print "1_data set/blotch/blotch ("+str(i)+").JPG"
	if(i<=68): image = cv2.imread("1_data set/blotch/blotch ("+str(i)+").JPG")
	else: image = cv2.imread("1_data set/blotch/blotch ("+str(i)+").jpg")

	segments = slic(img_as_float(image), n_segments = 20, sigma = 5)



	# show the output of SLIC
	fig = plt.figure("Superpixels")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
	plt.axis("off")
	plt.show()

	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype = "uint8")
		mask[segments == segVal] = 255

		# show the masked region
		# cv2.imshow("Mask", mask)
		cv2.imshow("Applied"+str(i), cv2.bitwise_and(image, image, mask = mask))
		# gray = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_BGR2GRAY)
		# lbp_image=local_binary_pattern(gray,8,2,method='nri_uniform')
		# histogram=scipy.stats.itemfreq(lbp_image.ravel())
	features=[]
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	a = raw_input("Enter the disease segment numbers:")
	b = a.split()
	
	for x in range(0,len(b)):
		b[x]=int(b[x])

	for (i, segVal) in enumerate(np.unique(segments)):
		print i
		mask = np.zeros(image.shape[:2], dtype = "uint8")
		mask[segments == segVal] = 255
		
		gray = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_BGR2GRAY)
		lbp_image=local_binary_pattern(gray,8,2,method='uniform')
		lbp_image = np.array(lbp_image)
		unique, counts = np.unique(lbp_image, return_counts=True)
		# print np.asarray((unique, counts)).T
		# x= raw_input("")
		histogram = np.asarray((unique, counts)).T #=scipy.stats.itemfreq(lbp_image.ravel())

		# features.append(histogram)
		if(len(histogram)>=8):
			mssg=""
			if(i in b):
				mssg=mssg+"2"
				print "ho pai"
			else: mssg=mssg+"1"
			# if(c==ord('o')):
			# 	mssg = mssg + "1"
			# 	print c
			# elif(c=='p'):
			# 	mssg=mssg+"2"
			# elif(c=='['):
			# 	mssg = mssg + "3"
			# elif(c==']'):
			# 	mssg = mssg + "4"
			for j in range(0,len(histogram)):
				mssg = mssg+" "+str(j+1)+":"+str(histogram[j][1])
			mssg=mssg+"\n"
			f1.write(mssg)

f1.close()