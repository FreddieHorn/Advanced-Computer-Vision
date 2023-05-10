from Tree import DecisionTree
import numpy as np
import json
import time


class Forest:
    def __init__(self, n_classes, n_trees=1, integral_images=[], labels=[], tree_param=[], bagging_fraction=1, mode='train'):

        self.tree_param = tree_param
        self.n_classes = n_classes
        self.ntrees = n_trees
        self.trees = []

        num_samples = len(integral_images)
        for i in range(n_trees):
            # Randomly only use a fraction of bagging_fraction of the training set
            sample_ids = np.random.choice(np.array(range(num_samples)), size=int(bagging_fraction * num_samples), replace=True)
            self.trees.append(DecisionTree(n_classes, integral_images[sample_ids], labels[sample_ids], tree_param, mode))

    # Function to create ensemble of trees
    # Should return a trained forest with n_trees
    def train(self):
        for i, tree in enumerate(self.trees):
            print("TREE NR. ", i)
            tree.train()
            

    # Function to apply the trained Random Forest on a test image
    # should return predicted class the test image
    def predict(self, image):
        prob = np.zeros(self.n_classes, dtype=float)
        for i, tree in enumerate(self.trees):
            prob += tree.predict(image)
        prob /= self.ntrees
        return prob


    # feel free to add any helper functions