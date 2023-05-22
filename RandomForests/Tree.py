import numpy as np
from Node import Node
import cv2
from tqdm import tqdm # just for progress bar visualization

def integral_image(img):
    intImg = np.zeros((img.shape))

    for y, row in enumerate(intImg): # for each row
        for x, _ in enumerate(row): # for each column
            if x > 0 and y > 0:
                intImg[y,x] = img[y,x] + intImg[y,x-1] + intImg[y-1,x] - intImg[y-1,x-1]
            elif y > 0 and x == 0:
                intImg[y,x] = img[y,x] + intImg[y-1,x]
            elif x > 0 and y == 0:
                intImg[y,x] = img[y,x] + intImg[y,x-1]
            elif x == 0 and y == 0:
                intImg[y,x] = img[y,x]
    return intImg


def patch_average(intImg, patch_center, patch_size=(15,15)):
    max_y = intImg.shape[0]-1
    max_x = intImg.shape[1]-1

    # assert(patch_size[0] > 0, "patchsize wrong!!!")
    # assert(patch_size[1] > 0, "patchsize wrong!!!")
    # assert(patch_size[0] <= max_y, "patchsize wrong!!!")
    # assert(patch_size[1] <= max_x, "patchsize wrong!!!")

    patch_size_y = patch_size[0]
    patch_size_x = patch_size[1]

    half_patch_size = np.floor(patch_size/2).astype(int)
    top_left = patch_center - (half_patch_size[0], half_patch_size[1]) - (1,1)
    bot_right = patch_center + (half_patch_size[0], half_patch_size[1])
    top_right = patch_center + (-half_patch_size[0], half_patch_size[1]) + (-1,0)
    bot_left = patch_center + (half_patch_size[0], -half_patch_size[1]) + (0,-1)

    if top_left[0] >= 0 and top_left[1] >= 0:
        intImg_top_left = intImg[top_left[0], top_left[1]]
    else:
        intImg_top_left = 0
        if top_left[0] < 0:
            patch_size_y = bot_left[0]
        if top_left[1] < 0:
            patch_size_x = top_right[1]

    if bot_right[0] <= max_y and bot_right[1] <= max_x:
        intImg_bot_right = intImg[bot_right[0], bot_right[1]]
    else:
        intImg_bot_right = 0
        if bot_right[0] > max_y:
            patch_size_y = max_y+1-top_right[0]
        if bot_right[1] > max_x:
            patch_size_x = max_x+1-bot_left[1]

    if top_right[0] >= 0 and top_right[1] <= max_x:
        intImg_top_right = intImg[top_right[0], top_right[1]]
    else:
        intImg_top_right = 0
        if top_right[0] < 0:
            patch_size_y = bot_right[0]+1
        if top_right[1] > max_x:
            patch_size_x = max_x+1-top_left[1]

    if bot_left[0] <= max_y and bot_left[1] >= 0:
        intImg_bot_left = intImg[bot_left[0], bot_left[1]]
    else:
        intImg_bot_left = 0
        if bot_left[0] > max_y:
            patch_size_y = max_y+1-top_left[0]
        if bot_left[1] < 0:
            patch_size_x = bot_right[1]+1


    area = patch_size_y * patch_size_x
    # assert(area>0, "AREA NOT 0!")
    avg = (intImg_bot_right - intImg_top_right - intImg_bot_left + intImg_top_left)/area
    # assert(np.array(avg).shape == (3,0), np.array(avg).shape)
    return avg


class DecisionTree:
    def __init__(self, n_classes, integral_images, labels, tree_param, mode):

        if mode == 'train':
            self.integral_images = integral_images
            self.labels = np.asarray([int(i) for i in labels])
            self.depth = tree_param['depth']
            # self.num_pixel_locations = tree_param['num_pixel_locations']
            # self.random_color_values = tree_param['random_color_values']
            # self.num_patch_sizes = tree_param['num_patch_sizes']
            # self.num_thresholds = tree_param['num_thresholds']
            self.minimum_samples_at_leaf = tree_param['minimum_samples_at_leaf']
            # self.classes = tree_param['classes']

        self.nodes = []
        self.n_classes = n_classes

    # Function to train the tree
    # returns a trained tree with provided tree param
    def train(self):
        IMG_SHAPE = self.integral_images[0].shape

        # # sample random splitting function parameters for the splitting functions
        # # (this is where each tree is unique)
        # q_1s = np.random.randint([0,0], [IMG_SHAPE[0], IMG_SHAPE[1]], (self.num_pixel_locations, 2))
        # q_2s = np.random.randint([0,0], [IMG_SHAPE[0], IMG_SHAPE[1]], (self.num_pixel_locations, 2))
        # color_channels = [0,1,2] # why should we sample 10 times?? As there can only be 3 channels, it makes no sense
        # patch_sizes = np.random.randint([0,0], [IMG_SHAPE[0], IMG_SHAPE[1]], (self.num_patch_sizes, 2))
        # thresholds = np.random.randint(-255, 255, self.num_thresholds)

        # params = [(q1, q2, c, ps, tau) for q1 in q_1s for q2 in q_2s for c in color_channels for ps in patch_sizes for tau in thresholds]

        # First create the root node
        self.root = Node(list(range(self.integral_images.shape[0])))
        self.root.depth = 0
        # put root node into current split nodes that need to be worked through
        split_nodes = [self.root]

        # for each working node, create a random set of splitting functions 
        # and among these choose the splitting function that maximizes information gain
        # For the two resulting nodes, check if any is a leaf node due to minimum number of samples,
        # otherwise put the node into the working node set

        while len(split_nodes) != 0:

            node = split_nodes.pop(-1)

            node_img_ids = node.training_image_ids


            best_params = None
            best_params_IG = -99999999
            best_params_positive_ids = []
            print("Processing new split node...")
            for i in tqdm(range(1000)):
                # actually sampling 1000 random parameters
                # I am NOT doing it exactly as in the task, because that would take way too long.
                # Here I just randomly sample one possible parameter tuple
                q1 = np.random.randint([0,0], [IMG_SHAPE[0], IMG_SHAPE[1]])
                q2 = np.random.randint([0,0], [IMG_SHAPE[0], IMG_SHAPE[1]])
                c = np.random.choice([0,1,2])
                ps = np.random.randint([1,1], [IMG_SHAPE[0], IMG_SHAPE[1]])
                tau = np.random.randint(-255, 256)
                # (q1, q2, c, ps, tau) = params[np.random.randint(len(params))]


                # Collect all positive samples
                positive_ids = self.getFeatureResponses(node_img_ids, (q1, q2, c, ps, tau))
                negative_ids = list(set(node_img_ids) - set(positive_ids))
                
                # Calculate information gain of this split
                ig = self.get_information_gain(self.compute_entropy(positive_ids),
                                          self.compute_entropy(negative_ids),
                                          self.compute_entropy(node_img_ids),
                                                               len(node_img_ids),
                                                               len(positive_ids),
                                                               len(negative_ids))
                # update best params if it's the best information gain so far
                if ig > best_params_IG:
                    best_params = (q1, q2, c, ps, tau)
                    best_params_IG = ig
                    best_params_positive_ids = positive_ids

            best_params_negative_ids = list(set(node_img_ids) - set(best_params_positive_ids))
            print("Best IG: ", np.round(best_params_IG,2), " with params ", best_params)
            # create the two child nodes
            node_left = Node(best_params_positive_ids)
            node_right = Node(best_params_negative_ids)
            node.set_as_SplitNode(node_left, node_right, best_params)

            # if a child should already be a leaf, make it a leaf, otherwise append it to split_nodes

            if node.depth == self.depth \
                or len(best_params_positive_ids) < self.minimum_samples_at_leaf \
                or np.all(self.labels[node_left.training_image_ids] == self.labels[node_left.training_image_ids][0]):
                node_left.set_as_leafNode(self.labels[node_left.training_image_ids], list(range(self.n_classes)))
            else:
                split_nodes.append(node_left)

            if node.depth == self.depth \
                or len(best_params_negative_ids) < self.minimum_samples_at_leaf \
                or np.all(self.labels[node_right.training_image_ids] == self.labels[node_right.training_image_ids][0]):
                node_right.set_as_leafNode(self.labels[node_right.training_image_ids], list(range(self.n_classes)))
            else:
                split_nodes.append(node_right)


    # Function to predict probabilities for single image
    # should return predicted class distribution in the test image
    def predict(self, image):
        intImage = integral_image(image)
        done = False

        curNode = self.root
        while not done:
            
            if self.getFeatureResponse(intImage, curNode.feature):
                curNode = curNode.leftChild
            else:
                curNode = curNode.rightChild

            if curNode.type == "leaf":
                done = True
        return curNode.probabilities

    # Function to get feature response for a random color and random locations
    # should return feature response for the image
    def getFeatureResponse(self, integralimage, feature):
        (q1, q2, c, ps, tau) = feature
        patch_average1 = patch_average(integralimage, q1, ps)
        patch_average2 = patch_average(integralimage, q2, ps)

        return patch_average1[c] - patch_average2[c] < tau
  

    # Function to get feature response for a random color and random locations
    # should return feature response for the image
    def getFeatureResponses(self, img_ids, feature):
        positive_ids = []
        for i in img_ids:
            if self.getFeatureResponse(self.integral_images[i], feature):
                positive_ids.append(i)

        return positive_ids
    


    # # Function to get left/right split given feature responses and a threshold
    # # provide your implementation
    # # should return left/right split
    # def getsplit(self, responses, threshold):
    #     pass

    # Function to compute entropy over incoming class labels
    def compute_entropy(self, ids):
        if len(ids) == 0:
            return 0
        p_i = np.array([np.sum(np.where(self.labels[ids] == i, 1, 0)) for i in range(self.n_classes)]).astype(float)
        p_i /= len(ids)

        # assert(np.sum(p_i)==1, str(p_i) + " is no distribution!")
        s = 0
        for p in p_i:
            if p != 0:
                s += -p * np.log2(p)

        return s

    # Function to measure information gain for a given split
    def get_information_gain(self, Entropyleft, Entropyright, EntropyAll, Nall, Nleft, Nright):
        return EntropyAll - ((Nleft/Nall)*Entropyleft + (Nright/Nall)*Entropyright)

    # # Function to get the best split for given images with labels
    # # provide your implementation
    # # should return left split, right split, feature (loc1, loc2, patch_size, color and threshold)
    # def best_split(self, ids):
    #     pass

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
    