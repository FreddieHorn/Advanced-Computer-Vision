import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from SVM import support_vector_machine
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
# Global constants

# crop/patch dimensions for the training samples
WIDTH = 64
HEIGHT = 128

win_size = (WIDTH, HEIGHT)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9

hog = cv.HOGDescriptor(win_size, block_size, block_stride,
                        cell_size, num_bins)

num_negative_samples = 10 # number of negative samples per image
train_hog_path = 'train_hog_descs' # the file to which you save the HOG descriptors of every patch

path_train =  './data/task_data/train/' #CHANGE PATHS DEPENDING ON THE WORKING DIR!
path_test = './data/task_data/test/'

hog_descriptor_size = 3780 # the feature dimension of a hog descriptor for a single block

colorspace = 1 # 1 for color, 0 for gray
#***********************************************************************************
# draw a bounding box in a given image
# Parameters:
# im: The image on which you want to draw the bounding boxes
# detections: the bounding boxes of the detections (people)
# returns None

#slightly changed the img reading functions from sheet2
def read_txt_file(filename):
    img_list = []
    with open(filename, "r") as infile: 
        for line in infile:
            line = line.strip()
            img_list.append(line) #value denotes a positive sample - 1 or a negative one - 0
    return img_list

def extract_img_data(img_list, DIR, positive): #sample_type is either 1 - positive, 0 - negative
    X = []
    Y_labels = []
    if positive:
        for i, img_path in enumerate(img_list):
            try:
                path = os.path.join(DIR, img_path)
                img = cv.imread(path, cv.IMREAD_COLOR)
                center = img.shape
                x = center[1]/2 - WIDTH/2
                y = center[0]/2 - HEIGHT/2
                crop_img = img[int(y):int(y+HEIGHT), int(x):int(x+WIDTH), :]
                
                X.append(hog.compute(crop_img, (16, 16)))
                print(i)
            except Exception as e: # catch "broken" images where width or height is 0, therefore cannot be resized. 
                print(str(e))
        Y_labels = np.ones(len(X))
        return np.array(X), Y_labels
    else:
        X = []
        for i, img_path in enumerate(img_list):
            try:
                path = os.path.join(DIR, img_path)
                img = cv.imread(path, cv.IMREAD_COLOR)
                patches = extract_patches(img)
                X.extend(patches)
            except Exception as e: # catch "broken" images where width or height is 0, therefore cannot be resized. 
                print(str(e))
        Y_labels = np.zeros(len(X))
        return np.array(X), Y_labels


def extract_patches(image, num_patches=10, patch_size=(128, 64)):
    image_height, image_width = image.shape[:2]
    patch_height, patch_width = patch_size

    patches = []

    for _ in range(num_patches):
        valid_patch = False

        while not valid_patch:
            # Generate random patch coordinates
            top = np.random.randint(0, image_height - patch_height + 1)
            left = np.random.randint(0, image_width - patch_width + 1)
            bottom = top + patch_height
            right = left + patch_width

            # Check if patch exceeds image boundaries
            if bottom <= image_height and right <= image_width:
                # Extract the patch from each channel
                patch = image[top:bottom, left:right, :]

                patches.append(hog.compute(patch, (16, 16)))
                valid_patch = True

    return patches

def RBF_embed(X, C, sigma):
    Z = np.zeros((X.shape[0], C.shape[0]))

    # # for each row of X and each row of C, calculate the RBF similarity
    # for i, x_i in enumerate(X):
    #     for j, c in enumerate(C):
    #         Z[i,j] = np.exp(-np.dot(x_i-c, x_i-c)/sigma**2)


    # write in matrix multiplication by broadcasting:
    # broadcast C and X to be of a common shape (num_samples, num_clusters, num_features)
    # meaning X is broadcasted along the cluster-axis and C is broadcasted along the sample-axis
    # so all cluster vectors will be substracted from all sample vectors
    # and after that map each of these vectors to its squared norm (<=>dot product), rest are element wise operations
    Z = np.exp(-((np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis,:,:], axis=2)**2)/(sigma**2)))

    # I measured performance and found that the broadcasting method was (only) about twice as fast as the explicit looping

    return Z


def task1_1():

      # TODO: Create a HOG descriptor object to extract the features from the set of positive and negative samples 

    # positive samples: Get a crop of size 64*128 at the center of the image then extract its HOG features
    # negative samples: Sample 10 crops from each negative sample at random and then extract their HOG features
    # In total you should have  (x+10*y) training samples represented as HOG features(x=number of positive images, y=number of negative images),
    # save them and their labels in the path train_hog_path and train_labels in order to load them in section 3 

    print('Task 1.1 - Extract HOG features')


    # Load image names
    print(os.getcwd())
    filelist_train_pos = path_train + 'filenamesTrainPos.txt'
    filelist_train_neg = path_train + 'filenamesTrainNeg.txt'

    img_pos_list = read_txt_file(filelist_train_pos)
    img_neg_list = read_txt_file(filelist_train_neg)

    X_train_pos, Y_train_pos = extract_img_data(img_pos_list, path_train+"/pos/", positive=True) 
    X_train_neg, Y_train_neg = extract_img_data(img_neg_list, path_train+"/neg/", positive=False)

    X_test_pos, Y_test_pos = extract_img_data(img_pos_list, path_test+"/pos/", positive=True) 
    X_test_neg, Y_test_neg = extract_img_data(img_neg_list, path_test+"/neg/", positive=False)

  #  X_train_pos, Y_train_pos = X_train_neg[:200,:], Y_train_neg[:200] # my computer cannot handle big training data
  #  X_train_neg, Y_train_neg = X_train_neg[:600,:], Y_train_neg[:600] # my computer cannot handle big training data
    # okay only X_train_pos and X_train_neg samples are needed for SVM training
    X_train = np.append(X_train_pos, X_train_neg, axis=0) #order of elements is maintained 
    Y_train = np.append(Y_train_pos, Y_train_neg, axis=0)


 #   X_test_pos, Y_test_pos = X_test_neg[:100,:], Y_train_neg[:100] # my computer cannot handle big training data
  #  X_test_neg, Y_test_neg = X_test_neg[:500,:], Y_train_neg[:500] # my computer cannot handle big training data

    X_test = np.append(X_test_pos, X_test_neg, axis=0) #order of elements is maintained 
    Y_test = np.append(Y_test_pos, Y_test_neg, axis=0)

    del X_train_pos
    del Y_train_pos
    del X_train_neg
    del Y_train_neg
    del X_test_pos
    del Y_test_pos
    del X_test_neg
    del Y_test_neg

    idx = np.random.permutation(len(X_train)) #shuffling the data
    X_train, Y_train = X_train[idx], Y_train[idx]

    svma = svm.SVC()
    # svm = support_vector_machine(features = X_train.shape[1],kernel="gaussian")
    svma.fit(X_train, Y_train)

    y_pred = svma.predict(X_test)

    print("Accuracy: "+str(accuracy_score(Y_test, y_pred)))

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