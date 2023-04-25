import numpy as np
from numpy.linalg import inv
from sklearn.cluster import KMeans

##############################################################################################################
#Auxiliary functions for Regression
##############################################################################################################
#returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:,:2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:,2:]), axis=1)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
#returns regression coefficients w ((1+num_features)*target_dims),
#  so w is what theta is in the lecture and Y is what w is in the lecture
def lin_reg(X, Y):
    return np.linalg.inv(X@X.T)@X@Y # just the formula derived in the lecture

#takes features with bias X (num_samples*(1+num_features)), target Y (num_samples*target_dims) and regression coefficients w ((1+num_features)*target_dims)
#returns the ratio of mean square error to variance of target prediction separately for each target dimension
def test_lin_reg(X, Y, w):
    pred_Y = X.T@w # predict what Y would be according to our model

    #calculate performance measures
    MSE = np.square(Y-pred_Y).mean(axis=0)
    var = np.var(Y, axis=0)

    return MSE/var

#takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and stdev of RBF sigma
#returns matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)
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


############################################################################################################
#Linear Regression
############################################################################################################

def run_lin_reg(X_tr, Y_tr, X_te, Y_te):
    X_tr = X_tr.T
    X_te = X_te.T

    w = lin_reg(X_tr, Y_tr) # train
    ratio = test_lin_reg(X_te, Y_te, w) # test, evaluate

    print("MSE/Var linear regression MSE to variance ratios are: ", ratio[0], " and ", ratio[1])

############################################################################################################
#Dual Regression
############################################################################################################

def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):

    # split train part into training and validation sets
    X_tr_part = X_tr[tr_list]
    X_val_part = X_tr[val_list]

    Y_tr_part = Y_tr[tr_list]
    Y_val_part = Y_tr[val_list]


    opt_sigma = 0
    opt_sigma_ratio = 99999999
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)
        
        # train
        # usually the dual parameters we get is np.linalg.inv(X.T@X)@Y
        # but since we use a kernel replacing the dot products, we will use np.linalg.inv(K)@Y

        K = RBF_embed(X_tr_part, X_tr_part, sigma)
        w = np.linalg.inv(K)@Y_tr_part #kernelized form of: w = np.linalg.inv(X_tr_part@X_tr_part.T)@Y_tr_part

        # test on validation set
        # Here the lecture was misleading/wrong I think (VERY confusing!!!)
        # In the inference equation they suggested X.T@X@psi instead of X.T@phi = X.T@X_train@psi !!!!!!
        pred_Y = RBF_embed(X_val_part, X_tr_part, sigma)@w #kernelized form of: pred_Y = X_val_part@X_tr_part.T@w

        #calculate performance measures
        MSE = np.square(Y_val_part-pred_Y).mean(axis=0)
        var = np.var(Y_val_part, axis=0)

        ratio = MSE/var

        print('MSE/Var dual linear regression MSE to variance ratios for val sigma='+str(sigma), " are ", ratio[0], " and ", ratio[1])

        if ratio[0] < opt_sigma_ratio:
            opt_sigma = sigma
            opt_sigma_ratio = ratio[0]

    # evaluate performance on test data, but first trained on ALL training data (including validation data)

    # again, first train
    K = RBF_embed(X_tr, X_tr, opt_sigma)
    w = np.linalg.inv(K)@Y_tr

    # predict on test set
    pred_Y = RBF_embed(X_te, X_tr, opt_sigma)@w

    # calculate performance measures
    MSE = np.square(Y_te-pred_Y).mean(axis=0)
    var = np.var(Y_te, axis=0)

    ratio = MSE/var
    print('best MSE/Var dual regression on test set for optimal test sigma='+str(opt_sigma), " are ", ratio[0], " and ", ratio[1])


    # Question answers:
    # What do you think about the val set proposed in the template? 
    #   I think it may be a bit bad because 
    #       when validating, the second target always had a much worse performance value than the first target (0.66 vs 4.26),
    #       however, when testing on all all training data on the final sigma, we get a much better performance on the second target (0.0024 vs 0.00059).
    #       This suggests that the validation data is much better for predicting the second target than the first target,
    #       while the training data is much better for predicting the first target than the second target,
    #       so the train and validation data are not from the exactly same distribution of data and so probably were not shuffled beforehand
    # What does the regressor become equivalent to if σ approaches 0? 
    #   As sigma approaches 0, the shape of the gaussian will peak and so in the limit k(xi,xj) != 0  iff xi=xj, ie. iff i=j
    #   so the kernel matrix will be proportional to the identity matrix, meaning that we overfit to the examples that we learned from
    #      
    # What happens to the regression if σ approaches inﬁnity?
    #   As sigma approaches infinity, the kernel similarity k(xi,xj) will approach 1 for all xi,xj.
    #   Then the kernel is useless (replacing dot product by 1 is not helpful, it basically neglects all the data that we have (underfitting))


############################################################################################################
#Non Linear Regression
############################################################################################################
def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    # split train part into training and validation sets
    X_tr_part = X_tr[tr_list]
    X_val_part = X_tr[val_list]

    Y_tr_part = Y_tr[tr_list]
    Y_val_part = Y_tr[val_list]

    opt_sigma = 0
    opt_num_clusters = 0
    opt_ratio = 99999999
    for num_clusters in [10, 30, 100]:
        # use kmeans to find centers for the RBF kernels
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_tr_part)
        centers = kmeans.cluster_centers_
        
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)

            Z = RBF_embed(X_tr_part, centers, sigma) # contains the z_i in row i
            Z = np.hstack((np.ones((Z.shape[0], 1)), Z)) #  adding the 1 entries for bias
            Z = Z.T

            w = lin_reg(Z, Y_tr_part) # train

            Z_val = RBF_embed(X_val_part, centers, sigma)
            Z_val = np.hstack((np.ones((Z_val.shape[0], 1)), Z_val))
            Z_val = Z_val.T

            ratio = test_lin_reg(Z_val, Y_val_part, w) # evaluate

            print('MSE/Var non linear regression for val sigma='+str(sigma)+' val num_clusters='+str(num_clusters), " are ", ratio[0], " and ", ratio[1])

            if ratio[0] < opt_ratio:
                opt_sigma = sigma
                opt_num_clusters = num_clusters
                opt_ratio = ratio[0]

    # evaluate performance on test data, but first trained on ALL training data (including validation data)

    # again, first train
    Z = RBF_embed(X_tr, centers, sigma) # contains the z_i in row i
    Z = np.hstack((np.ones((Z.shape[0], 1)), Z)) #  adding the 1 entries for bias
    Z = Z.T
    w = lin_reg(Z, Y_tr) # train

    Z_test = RBF_embed(X_te, centers, sigma)
    Z_test = np.hstack((np.ones((Z_test.shape[0], 1)), Z_test))
    Z_test = Z_test.T

    ratio = test_lin_reg(Z_test, Y_te, w) # evaluate
    
    print('best MSE/Var non linear regression for test sigma='+str(opt_sigma)+' test num_clusters='+str(opt_num_clusters), " are ", ratio[0], " and ", ratio[1])

    # Result: This seems to be a worse approach than the previous approaches:
    #       Even though the validation was best with sigma=9 and num_clusters=10, resulting in ratios 0.39 and  5.57,
    #       the final test evaluation was much worse with ratios 19.30 and 16.5,
    #       Meaning that the clusters picked were good features for the validation set but bad ones for the test set?




####################################################################################################################################
#Auxiliary functions for classification
####################################################################################################################################
#returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]: 
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns log likelihood loss
def get_loss(X, Y, w):
    loss = 0
    for i in range(X.shape[0]):
        loss += Y[i] * np.log((1/(1+np.exp(-np.dot(w, X[i])))) + 1e-9) + (1-Y[i])*np.log((np.exp(-np.dot(w, X[i]))/(1+np.exp(-np.dot(w, X[i])))) + 1e-9) # + 1e-9 to avoid log(0)
    return -loss
#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns accuracy
def get_accuracy(X, Y, w):
    A = X@w
    S = 1/(1 + np.exp(-A))

    Y_pred = np.where(S > 1/2, 1, -1)

    num_matches = np.count_nonzero(Y_pred == Y)
    accuracy = num_matches / Y.shape[0]
    return accuracy


####################################################################################################################################
#Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):

    # SOMETHING is wrong but I can't find the error. there is no improvement in training at all.

    # insert 1 for each data point for the bias
    X_tr = np.hstack((np.ones((X_tr.shape[0],1)), X_tr))
    X_te = np.hstack((np.ones((X_te.shape[0],1)), X_te))

    # normalize data
    for i in range(X_tr.shape[0]):
        X_tr[i,:] = (X_tr[i,:] - np.mean(X_tr[i,:]))/np.std(X_tr[i,:])
    for i in range(X_te.shape[0]):
        X_te[i,:] = (X_te[i,:] - np.mean(X_te[i,:]))/np.std(X_te[i,:])


    w = np.random.normal(0, scale=1, size=(X_tr.shape[1]))


    print('classification with step size '+str(step_size))
    max_iter = 10000
    accuracies = []
    losses = []
    for step in range(max_iter):
        if step%100 == 0:
            accuracy = get_accuracy(X_tr, Y_tr, w)
            loss = get_loss(X_tr, Y_tr, w)
            accuracies.append(accuracy)
            losses.append(loss)
            # print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))


        # batch gradient descent...

        A = X_tr@w # multiply each training example with w, yielding a vector of activations of length of the training examples number
        S = 1/(1 + np.exp(-A)) # send the activations through the sigmoid function
        gradient = - X_tr.T@(S - Y_tr) # this efficiently implements the derivative of the log-likelihood

        w = w + step_size * gradient

        # I also tried stochastic gradient descent...
        # rand_i = np.random.choice(X_tr.shape[0])
        # x_i = X_tr[rand_i]

        # a = X_tr[rand_i]
        # s = 1/(1 + np.exp(-np.dot(a, w)))
        # sample_gradient = - (s - Y_tr[rand_i])*x_i
        # w = w + 0.1*step_size * sample_gradient

        
    accuracy = get_accuracy(X_te, Y_te, w)
    loss = get_loss(X_te, Y_te, w)
    print('test set loss='+str(loss)+' accuracy='+str(accuracy))

    import matplotlib.pyplot as plt
    # plot the loss and accuracy over the course of training
    plt.plot(range(len(losses)), losses)
    plt.title('Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(range(len(accuracies)), accuracies)
    plt.title('Accuracy over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

    #Q we should answer: Can your model get stuck in a local minimum and why?
    #A: No because the log likelihood that we maximize is a sum of logarithms, which are concave,
    # so they whole log likelhood is also concave, therefore it has a single maximum

def main():
    ####################################################################################################################################
    #Exercises
    ####################################################################################################################################
    ## Exercise 1
    
    Y_tr, X_tr = read_data_reg('data/regression_train.txt')
    Y_te, X_te = read_data_reg('data/regression_test.txt')

    ########################################################
    # #optional shuffling
    # from sklearn.utils import shuffle

    # # shuffles just training set
    # # X_tr, Y_tr = shuffle(X_tr, Y_tr, random_state=0)

    # # shuffles all data
    # len_tr = X_tr.shape[0]
    # len_test = X_te.shape[0]
    # X = np.vstack((X_tr, X_te))
    # Y = np.vstack((Y_tr, Y_te))
    # X_tr = X[:len_tr]
    # Y_tr = Y[:len_tr]
    # X_te = X[:len_test]
    # Y_te = Y[:len_test]
    ########################################################

    # run_lin_reg(X_tr, Y_tr, X_te, Y_te)

    # tr_list = list(range(0, int(X_tr.shape[0]/2)))
    # val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]))

    # run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
    # run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)


    # Exercise 2

    step_size = 1.0
    Y_tr, X_tr = read_data_cls('train')
    Y_te, X_te = read_data_cls('test')

    # # shuffles just training set
    from sklearn.utils import shuffle
    X_tr, Y_tr = shuffle(X_tr, Y_tr, random_state=0)

    run_classification(X_tr, Y_tr, X_te, Y_te, step_size)

if __name__ == '__main__':
    main()
