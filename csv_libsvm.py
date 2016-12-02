from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

f = open("rotten.txt","r")
# t = open("test.txt","r")
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
# normalize(X,norm='l2',axis=0)
# testX = []
# testy = []
# for l in t:
# 	b = l[:-1].split(",")
# 	b = [float(x) for x in b]
# 	testX.append(b[:-1])
# 	testy.append(int(b[-1]))
# normalize(testX,norm='l2',axis=0)
f.close()
# t.close()



clf = SVC(kernel = 'rbf', C = 1.0, gamma = 0.01)
scores = cross_val_score(clf, X, y, cv=5)
# clf.fit(X_train, y_train)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print clf.score(X_test, y_test)
# print
# print confusion_matrix(y_test, clf.predict(X_test))

