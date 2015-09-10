#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# clf = SVC(kernel='linear')
# clf = SVC(kernel='rbf')

clf = SVC(C=10000.0, kernel='rbf')

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

tO = time()
clf.fit(features_train, labels_train)
print 'training time:', round(time()-tO, 3), 's'

tO = time()
pred = clf.predict(features_test)
print 'predicting time:', round(time()-tO, 3), 's'

#print 'Accuracy:', accuracy_score(labels_test, pred)

#print 'class for element 10:', clf.predict(features_test[10])
#print 'class for element 26:', clf.predict(features_test[26])
#print 'class for element 50:', clf.predict(features_test[50])

count = 0
for element in pred:
	if element == 1:
		count += 1

print 'The number of test events that are predicted to be in the class 1:', count

#########################################################
"""
Results for SVC with linear kernel for original dataset:
training time: 166.343 s
predicting time: 16.053 s
Accuracy: 0.984072810011

Results for SVC with linear kernel for 1% of original dataset:
training time: 0.092 s
predicting time: 0.942 s
Accuracy: 0.884527872582

Results for SVC with RBF kernel for 1% of original dataset:
training time: 0.236 s
predicting time: 1.065 s
Accuracy: 0.616040955631

Results for SVC with RBF kernel, C = 10.0 for 1% of original dataset:
training time: 0.124 s
predicting time: 1.078 s
Accuracy: 0.616040955631

Results for SVC with RBF kernel, C = 100.0 for 1% of original dataset:
training time: 0.104 s
predicting time: 1.064 s
Accuracy: 0.616040955631

Results for SVC with RBF kernel, C = 1000.0 for 1% of original dataset:
training time: 0.096 s
predicting time: 1.007 s
Accuracy: 0.821387940842

Results for SVC with RBF kernel, C = 10000.0 for 1% of original dataset:
training time: 0.092 s
predicting time: 0.837 s
Accuracy: 0.892491467577

Results for SVC with RBF kernel, C = 10000.0 for original dataset:
training time: 110.399 s
predicting time: 10.171 s
Accuracy: 0.990898748578

Results for SVC with RBF kernel, C = 10000.0 for 1% of original dataset:
class for element 10: [1]
class for element 26: [0]
class for element 50: [1]

Results for SVC with RBF kernel, C = 10000.0 for original dataset:
The number of test events that are predicted to be in the class 1: 877

"""


