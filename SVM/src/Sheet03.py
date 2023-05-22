import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Global constants

# crop/patch dimensions for the training samples
width = 64
height = 128

num_negative_samples = 10 # number of negative samples per image
train_hog_path = 'train_hog_descs' # the file to which you save the HOG descriptors of every patch

path_train =  '../data/task_data/train/' #
path_test = '..data/task_data/test/'

hog_descriptor_size = 3780 # the feature dimension of a hog descriptor for a single block

colorspace = 1 # 1 for color, 0 for gray
#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding boxes of the detections (people)
# returns None


def task1_1():

      # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples 

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3 



    print('Task 1.1 - Extract HOG features')


    # Load image names
  
    filelist_train_pos = path_train + 'filenamesTrainPos.txt'
    filelist_train_neg = path_train + 'filenamesTrainNeg.txt'


def task1_2(): 
    print('Task 1.2 - Train SVM and predict confidence values')
      #Create 3 SVMs with different C values, train them with the training data and save them
      # then use them to classify the test images and save the results


# plotting precision recall
def task1_3():
    pass





if __name__ == "__main__":

    # Task 1,2,3 
    task1_1()
    trained_svm = task1_2()
    task1_3()

    # Task 4
    pass