import numpy as np
from Node import Node
import cv2


class DecisionTree:
    def __init__(self, n_classes, images, labels, tree_param, mode):

        if mode == 'train':
            self.samples = images
            self.labels = np.asarray([int(i) for i in labels])
            self.depth = tree_param['depth']
            self.num_pixel_locations = tree_param['pixel_locations']
            self.random_color_values = tree_param['random_color_values']
            self.num_patch_sizes = tree_param['num_patch_sizes']
            self.num_thresholds = tree_param['num_thresholds']
            self.minimum_samples_at_leaf = tree_param['minimum_samples_at_leaf']
            self.classes = tree_param['classes']

        self.nodes = []
        self.n_classes = n_classes

    # Function to train the tree
    # provide your implementation
    # should return a trained tree with provided tree param
    def train(self):
        pass

    # Function to predict probabilities for single image
    # provide your implementation
    # should return predicted class distribution in the test image
    def predict(self, image):
        pass

    # Function to get feature response for a random color and random locations
    # provide your implementation
    # should return feature response for the image
    def getFeatureResponse(self, images, feature):
        color, q1, q2 = feature
        integral_img = self.integral_image(images)
        return None


    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold, ids):
        q1, q2, s, c = responses
        X = self.samples[ids]
        integral = self.integral_image(X[:,:,c])
        idx1 = (self.patch_multiple_average(X, q1[0], q1[1], s) - self.patch_multiple_average(X, q2[0], q2[1], s) <= threshold)
        idx2 = np.logical_not(idx1)

        return idx1, idx2 #left/right split


    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, ids): #is ids a list? 
        SUM = 0
        for single_class in self.n_classes:
            occurances = self.labels[ids].count(single_class)
            n_labels = len(ids)
            SUM += - (occurances/n_labels) * np.log2(occurances/n_labels + 1e-6) # many sources claimed that it's log2. Lecture did not mention that
        return SUM

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - (Nleft/Nall) * Entropyleft - (Nright/Nall) * Entropyright

    # Function to get the best split for given images with labels
    # provide your implementation
    # should return left split, right split, feature (loc1, loc2, patch_size, color and threshold)
    def best_split(self, ids):
        Y = self.labels[ids]

        m = Y.size
        if m <= 1:
            return None, None
        best_ig = -1
        # best_idx, best_thr = None, None
        for _ in range(self.num_pixel_locations):
            for c in np.random.choice([0, 1, 2], size=self.random_color_values):
                for tau in np.linspace(0, 255, num=self.num_thresholds):
                    for s in np.arange(10,30, step=20/self.num_patch_sizes):
                        # random_img_id = np.random.choice(ids)
                        x1, y1 = (np.random.randint(low=0, high=self.samples.shape[0]), \
                                        np.random.randint(low=0, high=self.samples.shape[1]))
                        x2, y2 = (np.random.randint(low=0, high=self.samples.shape[0]), \
                                        np.random.randint(low=0, high=self.samples.shape[1]))
                        q1 = (x1,y1)
                        q2 = (x2,y2)
                        left, right = self.getsplit((q1,q2,s,c), tau, ids)
                        ent_left = self.compute_entropy(left)
                        ent_right = self.compute_entropy(right)
                        ent_all = self.compute_entropy(ids)
                        Nall = len(ids)
                        Nleft = len(left)
                        Nright = len(right)
                        ig = self.get_information_gain(ent_left, ent_right, ent_all, Nall, Nleft, Nright)
                        if ig > best_ig:
                            best_ig = ig
                            feature = (q1, q2, s, c, tau)
                            left_split = left
                            right_split = right

        return left_split, right_split, feature





    # feel free to add any helper functions
    def get_patch_sizes(self):
            p_sizes = []
            for i in range(self.num_patch_sizes):
                p_sizes.append(np.random.randint(low = 10, high=30))
            return p_sizes
    
    def patch_multiple_average(integral, x, y, size=15):
    # Ensure that the input is a numpy array
        integral = np.array(integral)

        # Calculate the coordinates of the four corners of the patch for each integral image
        x1 = x - size // 2
        y1 = y - size // 2
        x2 = x + size // 2
        y2 = y + size // 2

        # Calculate the sum of pixel values within the patch using the integral image for each image
        sum_patch = integral[x2, y2, :] - integral[x1-1, y2, :] - integral[x2, y1-1, :] + integral[x1-1, y1-1, :]

        # Calculate the number of pixels within the patch
        num_pixels = size ** 2

        # Calculate the average of pixel values within the patch for each image
        avg_patch = sum_patch / num_pixels

        return avg_patch

    def integral_multiple_image(images):
    # Ensure that the input is a numpy array
        images = np.array(images)

        # Initialize an array of zeros with the same shape as the input images
        integral = np.zeros_like(images, dtype=np.uint32)

        # Compute the first row and column of the integral image for each image
        integral[:, 0, :] = images[:, 0, :]
        integral[0, :, :] = images[0, :, :]

        # Compute the rest of the integral image for each image using the formula
        for i in range(1, images.shape[1]):
            for j in range(1, images.shape[0]):
                integral[j, i, :] = images[j, i, :] + integral[j-1, i, :] + integral[j, i-1, :] - integral[j-1, i-1, :]

        return integral

    def integral_image(self, image):
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

    def patch_average(self, integral, x, y, size=15, pad_image=True):
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
    