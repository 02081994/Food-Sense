# import the necessary packages
from skimage.feature import greycomatrix, greycoprops
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from texture import local_binary_pattern
from skimage.util import img_as_float
from skimage import data
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

def crop_image(img,tol):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]



def getHB(image,mask1):
	segimg1 = cv2.bitwise_and(image, image, mask = mask1)

	seghsv1 = cv2.cvtColor(segimg1, cv2.COLOR_BGR2HSV)

	segLab1 = cv2.cvtColor(segimg1, cv2.COLOR_BGR2Lab)


	mean1 =  np.mean(np.mean(seghsv1,axis=0),axis=0)[0]
	mean2 =  np.mean(np.mean(segLab1,axis=0),axis=0)[2]

	return mean1,mean2


def getFeatures(filename,image,label):

	if(image is None):
		print "Empty image"
	seghsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	segLab1 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

	mssg=""
	for x in range(0,3):
		mssg = mssg + str(np.mean(np.mean(seghsv1,axis=0),axis=0)[x])+","
	
	for x in range(0,3):
		mssg = mssg + str(np.mean(np.mean(segLab1,axis=0),axis=0)[x])+","

	mssg=mssg+str(label)+"\n"

	filename.write(mssg)


def getLBPFeatures(filename,image,label):

	if(image is None):
		print "Empty image"
	lbp_image=local_binary_pattern(image,8,2,method='uniform')
	lbp_image = np.array(lbp_image)
	unique, counts = np.unique(lbp_image, return_counts=True)

	histogram = np.asarray((unique, counts)).T #=scipy.stats.itemfreq(lbp_image.ravel())
	mssg=""
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
		mssg = mssg+str(histogram[j][1])+","
	mssg=mssg+str(label)+"\n"

	filename.write(mssg)

def getGLCMFeatures(image,xs,ys):

	if(image is None):
		print "Empty image"

	glcm = greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
	xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
	ys.append(greycoprops(glcm, 'correlation')[0, 0])

	# histogram = np.asarray((unique, counts)).T #=scipy.stats.itemfreq(lbp_image.ravel())
	# mssg=""
	# # if(c==ord('o')):
	# # 	mssg = mssg + "1"
	# # 	print c
	# # elif(c=='p'):
	# # 	mssg=mssg+"2"
	# # elif(c=='['):
	# # 	mssg = mssg + "3"
	# # elif(c==']'):
	# # 	mssg = mssg + "4"
	# mssg = mssg+str(xs) 
	# mssg=mssg+str(label)+"\n"

	# filename.write(mssg)

def getDistance(image,mask1,mask2):
	segimg1 = cv2.bitwise_and(image, image, mask = mask1)
	segimg2 = cv2.bitwise_and(image, image, mask = mask2)

	seghsv1 = cv2.cvtColor(segimg1, cv2.COLOR_BGR2HSV)
	seghsv2 = cv2.cvtColor(segimg2, cv2.COLOR_BGR2HSV)

	segLab1 = cv2.cvtColor(segimg1, cv2.COLOR_BGR2Lab)
	segLab2 = cv2.cvtColor(segimg2, cv2.COLOR_BGR2Lab)

	hsvmean1 =  np.mean(np.mean(seghsv1,axis=0),axis=0)[0]
	hsvmean2 =  np.mean(np.mean(seghsv2,axis=0),axis=0)[0]
	
	Labmean1 =  np.mean(np.mean(segLab1,axis=0),axis=0)[2]
	Labmean2 =  np.mean(np.mean(segLab2,axis=0),axis=0)[2]

	dist = ((hsvmean1-hsvmean2)**2 + (Labmean1-Labmean2)**2)**0.5

	return dist  



def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters
	
def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum



# f1 = open("rotten.txt","a",0)
# filters = build_filters() 

xs=[]
ys=[]

xr=[]
yr=[]

for i in range (1,31):

	print "1_data set/rot/rot ("+str(i)+").JPG"
	if(i<=63): image = cv2.imread("1_data set/rot/rot ("+str(i)+").JPG")
	else: image = cv2.imread("1_data set/rot/rot ("+str(i)+").jpg")

	image = cv2.resize(image,(320,240))
	segments = slic(img_as_float(image), n_segments = 20, sigma = 5)



	# show the output of SLIC
	fig = plt.figure("Superpixels")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments,color=(0,0,0),outline_color=(0,0,0)))
	plt.axis("off")
	plt.show()

	flag = []
	Masks = []
	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		# print "[x] inspecting segment %d" % (i)
		mask = np.zeros(image.shape[:2], dtype = "uint8")
		mask[segments == segVal] = 255

		# show the masked region
		# cv2.imshow("Mask", mask)
		if(getHB(image,mask)[0] not in flag):
			flag.append(getHB(image,mask)[0])
			adding = []
			for (j, segVal1) in enumerate(np.unique(segments)):
				
				if(i!=j):
					mask2 = np.zeros(image.shape[:2], dtype = "uint8")
					mask2[segments == segVal1] = 255
					# mask1=mask+mask2
					# segimg = cv2.bitwise_and(image, image, mask = mask1)
					# cv2.imshow("Applied"+str(i), segimg)
						
					if(getDistance(image,mask,mask2)<=1.5 and (getHB(image,mask2)[0] not in flag)):
						adding.append(mask2)
						# np.delete(segments,j)
						# print len(segments)
					# print str(j)+" --> "+str(getDistance(image,mask,mask2))
			for j in adding:
				mask=mask+j
				flag.append(getHB(image,j)[0])

			Masks.append(mask)
				
		# x= raw_input("")
		# histogram = np.asarray((unique, counts)).T #=scipy.stats.itemfreq(lbp_image.ravel())
		# n_bins = lbp_image.max() + 1
		
		
		# print glcm

		# gray = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_BGR2GRAY)
		# lbp_image=local_binary_pattern(gray,8,2,method='nri_uniform')
		# histogram=scipy.stats.itemfreq(lbp_image.ravel())
	# features=[]
	i=0
	for mask in Masks:
		print "[x] inspecting segment %d" % (i)
		segimg = cv2.bitwise_and(image, image, mask = mask)
		cv2.imshow("Applied"+str(i), segimg)
		i=i+1
		# segments = slic(img_as_float(segimg), n_segments = 5, sigma = 5)
		# fig = plt.figure("Superpixels")
		# ax = fig.add_subplot(1, 1, 1)
		# ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(segimg, cv2.COLOR_BGR2RGB)), segments,color=(1,0,0)))
		# plt.axis("off")
		# plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	Disease = raw_input("Enter the disease segment numbers:")
	Disease = Disease.split()

	Normal = raw_input("Enter the Normal segment numbers:")
	Normal = Normal.split()


	for x in range(0,len(Disease)):
		Disease[x]=int(Disease[x])

	for x in range(0,len(Normal)):
		Normal[x]=int(Normal[x])

	i=0
	for mask in Masks:
		M = cv2.moments(mask)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		if(i in Disease):
			segimg = image[cy-25:cy+25,cx-25:cx+25]
			# cv2.imshow("Applied"+str(0), segimg)
			# cv2.waitKey(0)
			res1 = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
			# getLBPFeatures(f1,res1,1)
			getGLCMFeatures(res1,xs,ys)

		if(i in Normal):
			print (cy,cx)
			print len(image)
			print len(image[0])
			segimg = image[cy-25:cy+25,cx-25:cx+25]
			# cv2.imshow("Applied"+str(0), segimg)
			# cv2.waitKey(0)
			res1 = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)			
			# getLBPFeatures(f1,res1,0)
			getGLCMFeatures(res1,xr,yr)

		i=i+1


	# xs = []
	# ys = []
	# for (i, segVal) in enumerate(np.unique(segments)):
	# 	if(i in b):
	# 		mask = np.zeros(image.shape[:2], dtype = "uint8")
	# 		mask[segments == segVal] = 255

	# 		h,B  = getHB(image,mask)
	# 		xs.append(h)
	# 		ys.append(B)
	# 		# print str(h)+" -- "+str(B)
	# # 		# fig, ((ax1,histogram1)) = plt.subplots(nrows=1, ncols=2,
	# #   #                                                      figsize=(9, 6))
	# # 		# plt.gray()
			
	# # 		gray = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_BGR2GRAY)
	# # 		gray = crop_image(gray,0)
	# # 		glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
	# 		# xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
	# 		# ys.append(greycoprops(glcm, 'correlation')[0, 0])
	
	# for (i, segVal) in enumerate(np.unique(segments)):
	# 	if(i not in b):
	# 		mask = np.zeros(image.shape[:2], dtype = "uint8")
	# 		mask[segments == segVal] = 255
	# 		h,B  = getHB(image,mask)
	# 		xs.append(h)
	# 		ys.append(B)
	# 		# fig, ((ax1,histogram1)) = plt.subplots(nrows=1, ncols=2,
	#   #                                                      figsize=(9, 6))
	# 		# plt.gray()
			
	# 		gray = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask),cv2.COLOR_BGR2GRAY)
	# 		gray = crop_image(gray,0)
	# 		glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
	# 		xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
	# 		ys.append(greycoprops(glcm, 'correlation')[0, 0])
	# 	# unique, counts = np.unique(glcm, return_counts=True)
		# print np.asarray((unique, counts)).T
		# gray = ~gray
		# lbp_image=local_binary_pattern(gray,8,2,method='uniform')
		
		# lbp_image = np.array(lbp_image)
		# histogram = np.asarray((unique, counts)).T
		# # histogram1.hist(histogram, normed=True, bins=n_bins, range=(0, n_bins),facecolor='0.5')
		# # # features.append(histogram)
		# # plt.show()
		# if(len(histogram)>=):
		# 	mssg=""
		# 	if(i in b):
		# 		mssg=mssg+"2"
		# 		print "ho pai"
		# 	else: mssg=mssg+"1"
		# 	# if(c==ord('o')):
			# 	mssg = mssg + "1"
			# 	print c
			# elif(c=='p'):
			# 	mssg=mssg+"2"
			# elif(c=='['):
			# 	mssg = mssg + "3"
			# elif(c==']'):
			# 	mssg = mssg + "4"
			# for j in range(0,len(histogram)):
			# 	mssg = mssg+" "+str(j+1)+":"+str(histogram[j][1])
			# mssg=mssg+"\n"
	# 		# f1.write(mssg)
	# # # fig = plt.figure(figsize=(8, 8))
	# print len(xs)
	# print len(ys)
	# print len(b)
	# # # display original image with locations of patches

	# for each patch, plot (dissimilarity, correlation)
	# ax = fig.add_subplot(2,1)

plt.plot(xs[:], ys[:], 'ro',
        label='Rot')
plt.plot(xr[:], yr[:], 'bo',
        label='Normal')
# plt.set_xlabel('GLCM Dissimilarity')
# plt.set_ylabel('GLVM Correlation')
plt.legend()
# # plt.axis([0,250,0,250])
plt.show()

X=[]
y=[]

for i in range(len(xs)):
	X.append((xs[i],ys[i]))
	y.append(1)

for i in range(len(xr)):
	X.append((xr[i],yr[i]))
	y.append(0)

clf = SVC(kernel = 'rbf', C = 1.0, gamma = 0.01)
scores = cross_val_score(clf, X, y, cv=5)
# clf.fit(X_train, y_train)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print clf.score(X_test, y_test)

# f1.close()