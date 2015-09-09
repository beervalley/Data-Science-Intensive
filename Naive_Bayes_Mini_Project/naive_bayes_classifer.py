#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

tO = time()
clf.fit(features_train, labels_train)
print 'training time:', round(time()-tO, 3), 's'

tO = time()
pre = clf.predict(features_test)
print 'predicting time:', round(time()-tO, 3), 's'

from sklearn.metrics import accuracy_score

print 'accuracy:', accuracy_score(labels_test, pre)
#########################################################

"""
Results for Naive Beyes which can be used to compare with other algorithms for
SPEED and ACCURACY:

training time: 1.339 s
predicting time: 0.317 s
accuracy: 0.973833902162

"""


