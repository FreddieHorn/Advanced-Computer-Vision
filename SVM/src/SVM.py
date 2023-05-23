import numpy as np

class support_vector_machine:
    def __init__(self,C=150,features=2,sigma_sq=1,kernel="None"):
        self.C=C
        self.features=features
        self.sigma_sq=sigma_sq
        self.kernel=kernel
        self.weights=np.zeros(features)
        self.bias=0.
        
    def __similarity(self,x,l):
        return np.exp(-sum((x-l)**2)/(2*self.sigma_sq))

    def gaussian_kernel(self,x1,x):
        m=x.shape[0]
        n=x1.shape[0]
        op=[[self.__similarity(x1[x_index],x[l_index]) for l_index in range(m)] for x_index in range(n)]
        return np.array(op)
    def RBF(self, X):

        gamma = 1/(2*self.sigma_sq ** 2)
        # RBF kernel Equation
        K = np.exp(-gamma * np.sum((X - X[:,np.newaxis])**2, axis = -1))

        return K
    def RBF_embed(self, X, C):
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
        Z = np.exp(-((np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis,:,:], axis=2)**2)/(self.sigma_sq**2)))

    # I measured performance and found that the broadcasting method was (only) about twice as fast as the explicit looping

        return Z

    def loss_function(self,y,y_hat):
        sum_terms=1-y*y_hat
        sum_terms=np.where(sum_terms<0,0,sum_terms)
        return (self.C*np.sum(sum_terms)/len(y)+sum(self.weights**2)/2)

    def fit(self,x_train,y_train,epochs=1000,print_every_nth_epoch=100,learning_rate=0.01):
        y=y_train.copy()
        x=x_train.copy()
        self.initial=x.copy()
        
        assert x.shape[0]==y.shape[0] , "Samples of x and y don't match."
        assert x.shape[1]==self.features , "Number of Features don't match"
        
        if(self.kernel=="gaussian"):
            x=self.RBF_embed(x,x)
            m=x.shape[0]
            self.weights=np.zeros(m)

        n=x.shape[0]
        
        for epoch in range(epochs):
            y_hat=np.dot(x,self.weights)+self.bias
            grad_weights=(-self.C*np.multiply(y,x.T).T+self.weights).T
            
            for weight in range(self.weights.shape[0]):
                grad_weights[weight]=np.where(1-y_hat<=0,self.weights[weight],grad_weights[weight])
            
            grad_weights=np.sum(grad_weights,axis=1)
            self.weights-=learning_rate*grad_weights/n
            grad_bias=-y*self.bias
            grad_bias=np.where(1-y_hat<=0,0,grad_bias)
            grad_bias=sum(grad_bias)
            self.bias-=grad_bias*learning_rate/n
            if((epoch+1)%print_every_nth_epoch==0):
                print("--------------- Epoch {} --> Loss = {} ---------------".format(epoch+1, self.loss_function(y,y_hat)))
    
    def evaluate(self,x,y):
        pred=self.predict(x)
        pred=np.where(pred==0,1)
        diff=np.abs(np.where(y==0,1)-pred)
        return((len(diff)-sum(diff))/len(diff))

    def predict(self,x):
        if(self.kernel=="gaussian"):
            x=self.RBF_embed(x,self.initial)

        y_lmao = np.dot(x,self.weights)+self.bias
        return np.where(np.dot(x,self.weights)+self.bias>0,1,0)