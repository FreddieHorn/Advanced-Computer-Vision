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

    # for each row of X and each row of C, calculate the RBF similarity (I dunno how to write this in parallel vectorization)
    for i, x_i in enumerate(X):
        for j, c in enumerate(C):
            Z[i,j] = np.exp(-np.dot(x_i-c, x_i-c)/sigma**2)

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

    # # Now I define how I choose the RBF centers: Uniformly at random from the training data
    # num_clusters = 150 # How many clusters I chose
    # C = X_tr_part[np.random.choice(X_tr_part.shape[0], num_clusters)]

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
    print('MSE/Var dual regression on test set for optimal test sigma='+str(opt_sigma), " are ", ratio[0], " and ", ratio[1])

# ############################################################################################################
# #Non Linear Regression
# ############################################################################################################
# def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    
#     for num_clusters in [10, 30, 100]:
#         for sigma_pow in range(-5, 3):
#             sigma = np.power(3.0, sigma_pow)
#             print('MSE/Var non linear regression for val sigma='+str(sigma)+' val num_clusters='+str(num_clusters))
#             print(err_dual)

#     print('MSE/Var non linear regression for test sigma='+str(opt_sigma)+' test num_clusters='+str(opt_num_clusters))
#     print(err_dual)





# ####################################################################################################################################
# #Auxiliary functions for classification
# ####################################################################################################################################
# #returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
# def read_data_cls(split):
#     feat = {}
#     gt = {}
#     for category in [('bottle', 1), ('horse', -1)]: 
#         feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
#         feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
#         gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
#     X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
#     Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
#     return Y, X

# #takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# # Y must be from {-1, 1}
# #returns gradient with respect to w (num_features)
# def log_llkhd_grad(X, Y, w):

# #takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# # Y must be from {-1, 1}
# #returns log likelihood loss
# def get_loss(X, Y, w):

# #takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# # Y must be from {-1, 1}
# #returns accuracy
# def get_accuracy(X, Y, w):

# ####################################################################################################################################
# #Classification
# ####################################################################################################################################
# def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
#     print('classification with step size '+str(step_size))
#     max_iter = 10000
#     for step in range(max_iter):
#         if step%1000 == 0:
#             print('step='+str(step)+' loss='+str(loss)+' accuracy='+str(accuracy))

#     print('test set loss='+str(loss)+' accuracy='+str(accuracy))


def main():
    ####################################################################################################################################
    #Exercises
    ####################################################################################################################################
    ## Exercise 1
    
    Y_tr, X_tr = read_data_reg('data/regression_train.txt')
    Y_te, X_te = read_data_reg('data/regression_test.txt')


    run_lin_reg(X_tr, Y_tr, X_te, Y_te)

    tr_list = list(range(0, int(X_tr.shape[0]/2)))
    val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]))

    run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
    # run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)


    ## Exercise 2

    # step_size = 1.0
    # Y_tr, X_tr = read_data_cls('test')
    # Y_te, X_te = read_data_cls('test')
    # run_classification(X_tr, Y_tr, X_te, Y_te, step_size)

if __name__ == '__main__':
    main()
