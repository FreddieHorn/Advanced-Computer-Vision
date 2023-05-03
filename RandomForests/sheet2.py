from RandomForest import Forest
import matplotlib.pyplot as plt
import json
import cv2
from sklearn.metrics import confusion_matrix
import os #DELETE IT SO WE DONT LOSE POINTS. ONLY FOR CHECKING DIR
import numpy as np



IMAGE_WIDTH = 500
IMAGE_HEIGHT = 375
def read_txt_file(filename):
    dict = {}
    img_list = []
    label_list = []
    with open(filename, "r") as infile: 
        for i, line in enumerate(infile):
            if i == 0:
                numImages, numClasses = line.strip().split(" ")
                continue
            key, value = line.strip().split(" ")
            dict[key] = int(value)


    return dict, int(numImages), int(numClasses)

def extract_img_data(data_dict, DIR, size):
    X = np.empty((size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    Y_labels = []
    # for filename in os.listdir(DIR):
    #     # if filename.is_file():
    #     #     print(filename)
    #     # f = os.path.join(DIR, filename)
    #     # if os.path.isfile(f):
    #     #     print(filename)
    #     # if filename.startswith('.'):
    #     #     continue
    for i, (key, value) in enumerate(data_dict.items()):
        try:
            path = os.path.join(DIR,key)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
            X[i] = np.array(img)
            Y_labels.append(value)
        except Exception as e: # catch "broken" images where width or height is 0, therefore cannot be resized. 
            print(str(e))
    return X, np.array(Y_labels)

def main():
    # provide your implementation for the sheet 2 here
    print('Get current working directory : ', os.getcwd())
    train_dict, train_size, _ = read_txt_file("RandomForests/images/images/train_images.txt") #PATH IS DEPENDENT ON THE WORKING DIR!
    test_dict, test_size, _ = read_txt_file("RandomForests/images/images/test_images.txt") #PATH IS DEPENDENT ON THE WORKING DIR!
    X_train, Y_train = extract_img_data(train_dict, "RandomForests/images/images/train/", train_size)
    X_test, Y_test = extract_img_data(test_dict, "RandomForests/images/images/test/", test_size)

    x = 2
if __name__ == "__main__":
    main()