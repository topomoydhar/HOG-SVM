# Import the required modules
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import joblib
import argparse as ap
import glob
import os
from config import *

###########################
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
###########################

def trainer():

    args={"posfeat":'./pos_neg_features/pos', "negfeat": './pos_neg_features/neg','classifier':'LIN_SVM'}

    pos_feat_path =  args["posfeat"]
    neg_feat_path = args["negfeat"]

    # Classifiers supported
    clf_type = args['classifier']

    model_path = './svm_models/svm_gaussian.model'
    # model_path = 'svm.model'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)

    if clf_type == "LIN_SVM":
        clf = LinearSVC()
        print("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))

    elif clf_type == "RBF":
        # clf= SVC()
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(fds, labels)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

trainer()
