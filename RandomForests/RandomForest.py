from Tree import DecisionTree
import numpy as np
import json
import time


class Forest:
    def __init__(self, n_classes, n_trees=1, mode='train', images=[], labels=[], tree_param=[]):

        self.tree_param = tree_param
        self.n_classes = n_classes
        self.ntrees = n_trees
        self.trees = []
        for i in range(n_trees):
            self.trees.append(DecisionTree(n_classes, images, labels, tree_param, mode))

    # Function to create ensemble of trees
    # provide your implementation
    # Should return a trained forest with n_trees
    def create_forest(self):
        pass

    # Function to apply the trained Random Forest on a test image
    # provide your implementation
    # should return predicted class the test image
    def test(self, image):
        pass

    # feel free to add any helper functions