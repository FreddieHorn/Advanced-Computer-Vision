import numpy as np

class Node:
    def __init__(self, training_image_ids):
        self.depth = 0
        self.type = 'None'
        self.leftChild = None
        self.rightChild = None
        self.feature = None
        self.probabilities = []
        self.training_image_ids = training_image_ids # all images used to train this node

    # Function to create a new split node
    def set_as_SplitNode(self, leftchild, rightchild, feature):
        self.type = "split"
        self.leftChild = leftchild
        self.rightChild = rightchild
        self.feature = feature

        self.leftChild.depth = self.depth + 1
        self.rightChild.depth = self.depth + 1

    # Function to create a new leaf node
    def set_as_leafNode(self, labels, classes):
        
        self.type = "leaf"
        p_i = np.array([np.sum(np.where(labels == i, 1, 0)) for i in classes]).astype(float)
        p_i /= len(labels)
        self.probabilities = p_i

        print("new leaf has ", len(labels), " labels with distribution ", str(p_i), "!")

