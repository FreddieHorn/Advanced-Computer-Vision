import numpy as np

class Node:
    def __init__(self):

        self.type = 'None'
        self.leftChild = -1
        self.rightChild = -1
        self.feature = {'color': -1, 'location1': -1, 'location2': -1, 'patch_size':-1, 'th': -1}
        self.probabilities = []

    # Function to create a new split node
    # provide your implementation
    def create_SplitNode(self, leftchild, rightchild, feature):
        node = Node()
        node.type = 'split'
        node.leftChild = leftchild
        node.rightChild = rightchild
        node.feature = feature
        return node

    # Function to create a new leaf node
    # provide your implementation
    def create_leafNode(self, labels, classes):
        node = Node()
        node.type = 'leaf'
        node.probabilities = []
        for c in classes:
            node.probabilities.append(np.sum(labels == c) / len(labels))
        return node

    # feel free to add any helper functions
