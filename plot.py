from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np

f = open("lbp_features2.csv","r")
t = open("svm2","w")
xs = []
ys = []

total = 0
for l in f:
	b = l[:].split(",")
	b = [float(x) for x in b]
	if(b[2]==0.0):
		print "abc"
		total = total+1
		xs.append(b[0])
		ys.append(b[1])

f.close()
f = open("lbp_features2.csv","r")
total1 =0
for l in f:
	b = l[:].split(",")
	b = [float(x) for x in b]
	print b
	total1 = total1+1
	if(b[2]==1.0):
		print 
		xs.append(b[0])
		ys.append(b[1])

print total
print len(xs)
print len(ys)
# display original image with locations of patches

# for each patch, plot (dissimilarity, correlation)
# ax = fig.add_subplot(2,1)
plt.plot(xs[:total], ys[:total], 'go',
        label='blotch')
plt.plot(xs[total:], ys[total:], 'bo',
        label='Normal')
# plt.set_xlabel('GLCM Dissimilarity')
# plt.set_ylabel('GLVM Correlation')
plt.legend()
plt.show()

f.close()