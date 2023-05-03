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
        pass

    # Function to get left/right split given feature responses and a threshold
    # provide your implementation
    # should return left/right split
    def getsplit(self, responses, threshold):
        pass

    # Function to compute entropy over incoming class labels
    # provide your implementation
    def compute_entropy(self, ids):
        pass

    # Function to measure information gain for a given split
    # provide your implementation
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        pass

    # Function to get the best split for given images with labels
    # provide your implementation
    # should return left split, right split, feature (loc1, loc2, patch_size, color and threshold)
    def best_split(self, ids):
        pass

    # feel free to add any helper functions