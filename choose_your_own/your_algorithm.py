#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time

##1.- Using K-Nearest Neighbours
# Parameters
algorithmParamArray=['auto','ball_tree','kd_tree','brute']

for algorithmParam in algorithmParamArray:
    # Train Data
    print "*******Algortihm used:", algorithmParam
    print "----------Training Phase (KNeighbors)---------"
    t0 = time()
    clfKNN=KNeighborsClassifier(n_neighbors=10,algorithm=algorithmParam)
    clfKNN.fit(features_train, labels_train)
    print "training time (KNeighbors):", round(time()-t0, 3), "s"
    
    # Test Data
    print "----------Testing Phase (KNeighbors)---------"
    accuracy=clfKNN.score(features_test,labels_test)
    print "Accuracy (KNeighbors):", accuracy, '\n'
    try:
        prettyPicture(clfKNN, features_test, labels_test)
    except NameError:
        pass


##2.- Using Random Forest
# Parameters
#algorithmParamArray=['auto','ball_tree','kd_tree','brute']
algorithmParam='Random Forest' 
# Train Data
print "*******Algortihm used:", algorithmParam
print "----------Training Phase (Random Forest)----"
t0 = time()
clfRF=RandomForestClassifier(n_estimators=10)
clfRF.fit(features_train, labels_train)
print "training time (Random Forest):", round(time()-t0, 3), "s"

# Test Data
print "----------Testing Phase (Random Forest)----"
accuracy=clfRF.score(features_test,labels_test)
print "Accuracy (Random Forest):", accuracy, '\n'

try:
    prettyPicture(clfRF, features_test, labels_test)
except NameError:
    pass

##3.- Using AdaBoost (Decision Tree as Base Learner)
# Parameters
#algorithmParamArray=['auto','ball_tree','kd_tree','brute']
n_estimators_par=100 
# Train Data
print "*******Algortihm used:", " AdaBoost (Tree)"
print "----------Training Phase (AdaBoost)----"
t0 = time()
clfAB=AdaBoostClassifier(n_estimators=n_estimators_par)
clfAB.fit(features_train, labels_train)
print "training time (AdaBoost):", round(time()-t0, 3), "s"

# Test Data
print "----------Testing Phase (Random Forest)----"
accuracy=clfAB.score(features_test,labels_test)
print "Accuracy (AdaBoost):", accuracy, '\n'

try:
    prettyPicture(clfAB, features_test, labels_test)
except NameError:
    pass
