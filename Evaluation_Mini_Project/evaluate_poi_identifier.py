#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### training and testing
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# Calculate the predicted labels as POI
count = 0
for label in pred:
	if label == 1.:
		count += 1
		
print count

# Print the number of people in the test set
print len(pred)

# The evaluation metrcis for the prediction results
print "Accuracy:", clf.score(features_test, labels_test)
print "Accuracy:", accuracy_score(labels_test, pred)
print "Precision Score:", precision_score(labels_test, pred)
print "Recall Score:", recall_score(labels_test, pred)
print "F1 Score:", f1_score(labels_test, pred)


