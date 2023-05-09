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
    for i, (key, value) in enumerate(data_dict.items()):
        try:
            path = os.path.join(DIR,key)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
            X[i] = np.array(img)
            Y_labels.append(value)
        except Exception as e: # catch "broken" images where width or height is 0, therefore cannot be resized. But this situation 
            print(str(e))
    return X, np.array(Y_labels)

def integral_image(image):
        # Initialize an array of zeros with the same shape as the input image
        integral = np.zeros_like(image, dtype=np.uint32)

        # Compute the first row and column of the integral image
        integral[0, :] = image[0, :]
        integral[:, 0] = image[:, 0]

        # Compute the rest of the integral image using the formula
        for i in range(1, image.shape[0]):
            for j in range(1, image.shape[1]):
                integral[i, j] = image[i, j] + integral[i-1, j] + integral[i, j-1] - integral[i-1, j-1]

        return integral

def patch_average(integral, x, y, size=15, pad_image=True):
# Calculate the coordinates of the four corners of the patch
    pad_size = (size - 1) // 2

    # If pad_image is True, pad the image with zeros
    if pad_image:
        integral = np.pad(integral, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    x1 = x - pad_size
    y1 = y - pad_size
    x2 = x + pad_size
    y2 = y + pad_size

    # Calculate the sum of pixel values within the patch using the integral image
    sum_patch = integral[x2, y2] - integral[x1-1, y2] - integral[x2, y1-1] + integral[x1-1, y1-1]

    # Calculate the number of pixels within the patch
    num_pixels = size ** 2

    # Calculate the average of pixel values within the patch
    avg_patch = sum_patch / num_pixels

    return avg_patch
    

def main():
    # provide your implementation for the sheet 2 here
    print('Get current working directory : ', os.getcwd())
    train_dict, train_size, num_classes = read_txt_file("RandomForests/images/images/train_images.txt") #PATH IS DEPENDENT ON THE WORKING DIR!
    test_dict, test_size, _ = read_txt_file("RandomForests/images/images/test_images.txt") #PATH IS DEPENDENT ON THE WORKING DIR!
    X_train, Y_train = extract_img_data(train_dict, "RandomForests/images/images/train/", train_size)
    X_test, Y_test = extract_img_data(test_dict, "RandomForests/images/images/test/", test_size)

    example_integral  = integral_image(X_train[1][:,:,0])
    patch = patch_average(example_integral, 10, 10, 15, False)
    
    tree_params = {
        "depth" : 5,
        'num_patch_sizes' : 5,
        'num_thresholds' : 20,
        'random_color_values' : 10,
        'pixel_locations' : 100,
    }
    decision_tree = Forest(num_classes, 1, images=X_train, labels=Y_test, )

if __name__ == "__main__":
    main()