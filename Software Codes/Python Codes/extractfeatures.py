# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
import joblib
# To read file names
import argparse as ap
import glob
import os
from config import *

def feature_extractor():
    
    args = {}
    args["path"]='../CarData/CarData/TrainImages'
    args["descriptor"]="HOG"

    path = args["path"]

    cnt=0
   
    des_type = args["descriptor"]

    pos_feat_ph = './pos_neg_features/pos'
    neg_feat_ph = './pos_neg_features/neg'

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)


    for i in os.listdir(path):
      # print(i)
      cnt=cnt+1
      if os.path.isfile(os.path.join(path,i)) and 'pos' in i:
        # print(i)
        image_act_path=path+'/'+i
        im = imread(image_act_path, as_gray=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        fd_name = os.path.split(i)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
      
      elif os.path.isfile(os.path.join(path,i)) and 'neg' in i:
        image_act_path=path+'/'+i
        im = imread(image_act_path, as_gray=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize,transform_sqrt=transform_sqrt)
        fd_name = os.path.split(i)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)

    print("Positive features saved in {}".format(pos_feat_ph))

    print("Negative features saved in {}".format(neg_feat_ph))



    print("Completed calculating features from training images")
    print(cnt)

feature_extractor()

