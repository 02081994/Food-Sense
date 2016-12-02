from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


f = open("rotten.txt","r")
# t = open("training.txt","r")
X = []
y = []

for l in f:
	b = l[:-1].split(",")
	b = [float(x) for x in b]
	X.append(b[:-1])
	y.append(int(b[-1]))
	# print i
	# if (int(b[len(b)-1]) == 1 or int(b[len(b)-1]) == 2): 	
	#  	mssg = b[len(b)-1]
	#  	for x in range(0,len(b)-1):
	#  		mssg = mssg + " " + str(x+1)+":"+b[x]
	#  	mssg = mssg+"\n"
	# 	t.write(mssg)
	# i=i+1
# testX = []
# testy = []

# for l in t:
# 	b = l[:-1].split(",")
# 	b = [float(x) for x in b]
# 	testX.append(b[:-1])
# 	testy.append(int(b[-1]))

f.close()
# t.close()


pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
# testX = pca.transform(testX)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


clf = SVC(kernel = 'rbf', C = 10.0, gamma = 0.001)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print
print confusion_matrix(y_test, clf.predict(X_test))

eigenvalues = pca.explained_variance_

# for x in eigenvalues:
# 	print x

xs=[]
ys=[]

normal=0
for z in range(0,len(X)):
	if(y[z]==0):
		normal=normal+1
		xs.append(X[z][0])
		ys.append(X[z][1])

rotten=0
for z in range(0,len(X)):
	if(y[z]==1):
		rotten=rotten+1
		xs.append(X[z][0])
		ys.append(X[z][1])

for z in range(0,len(X)):
	if(y[z]==2):
		xs.append(X[z][0])
		ys.append(X[z][1])

plt.plot(xs[:normal], ys[:normal], 'ro',
        label='Normal')
plt.plot(xs[normal:normal+rotten], ys[normal:normal+rotten], 'bo',
        label='Rotten')
plt.plot(xs[normal+rotten:], ys[normal+rotten:], 'yo',
        label='Scab')
plt.legend()
plt.show()