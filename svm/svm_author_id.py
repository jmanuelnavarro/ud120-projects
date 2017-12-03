#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
from collections import Counter

#Slice trainign and testing data (delete this block for full training / testing data)
slice = 100
features_train=features_train[:len(features_train)/slice]
#features_test=features_test[:len(features_test)/slice]
labels_train=labels_train[:len(labels_train)/slice]
#labels_test=labels_test[:len(labels_test)/slice]

#Training parameters
c_param = 10000
gamma_param = 'auto'
print "training parameters: ", "C=",c_param, "Gamma=",gamma_param

#Training and testing for a linear kernel
print "__________starting training (linear)____________"
t0 = time()
clf_linear=svm.SVC(C=c_param, kernel='linear',gamma=gamma_param)
clf_linear.fit(features_train, labels_train)
print "training time (linear):", round(time()-t0, 3), "s"
print "accuracy for linear kernel:", clf_linear.score(features_test,labels_test)

#Training and testing for a rbf kernel
print "__________starting training (rbf)______________"
t0 = time()
clf_rbf=svm.SVC(C=c_param, kernel='rbf',gamma=gamma_param)
clf_rbf.fit(features_train, labels_train)
print "training time (rbf):", round(time()-t0, 3), "s"
print "accuracy for rfb kernel:", clf_rbf.score(features_test,labels_test)

#Print some prediction values
print "__________predicting some values______________"
idx_values = (10, 26, 50) #Assuming zero index list
for value in idx_values:
    print "predicting value ", value, ": ", features_test[[value]]
    print "predicted value: ", clf_rbf.predict(features_test[[value]])
    print "tested value: ", labels_test[value]

#How many are predicted as been writen by Chris?
print "__________counting predictions______________"
pred=clf_rbf.predict(features_test)
print "number of mails PREDICTED as writen by Chris (1) and Sarah (0): ",Counter(pred) 
print "number of mails ACTUALLY writen by Chris (1) and Sarah (0): ",Counter(labels_test) 

#########################################################


