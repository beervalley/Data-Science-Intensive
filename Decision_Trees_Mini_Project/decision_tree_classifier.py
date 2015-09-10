#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'The number of features:', len(features_train[0])
print 'Accuracy:', accuracy_score(labels_test, pred)

#########################################################

"""
The number of features: 3785
Results for Decision Tree Classifier with parameter min_samples_split equals 40
with 10% of the available features:
Accuracy: 0.978953356086

The number of features: 379
Results for Decision Tree Classifier with parameter min_samples_split equals 40
with 1% of the available features:
Accuracy: 0.966439135381

"""
