import numpy as np
import math

def naive_bayes(train_x, train_y, test_x):
    #compute the priori probability
    pY_ck_0=0
    pY_ck_1=0
    for i in range(0, train_y.shape[0]):
        if train_y[i]==-1:
            pY_ck_0+=1
        else:
            pY_ck_1+=1
    pY_ck_0/=train_y.shape[0]
    pY_ck_1/=train_y.shape[0]
    print(pY_ck_0,pY_ck_1)
    #compute the conditional probability
    pXj_ajl_Y=np.zeros((train_x.shape[1], 3, 2))
    for i in range(0, train_x.shape[0]):
        for j in range(0, train_x.shape[1]):
            pXj_ajl_Y[j, train_x[i][j]-1, math.floor((train_y[i]+1)/2)]+=1
    pXj_ajl_Y[:, :, 0]/=pY_ck_0*train_y.shape[0]
    pXj_ajl_Y[:, :, 1]/=pY_ck_1*train_y.shape[0]
    print(pXj_ajl_Y[:, :, 0])
    print(pXj_ajl_Y[:, :, 1])
    #compute the posterior probability
    poster_p_Y_ck_0=pY_ck_0
    poster_p_Y_ck_1=pY_ck_1
    for j in range(0, train_x.shape[1]):
        poster_p_Y_ck_0*=pXj_ajl_Y[j, test_x[j]-1, 0]
        poster_p_Y_ck_1*=pXj_ajl_Y[j, test_x[j]-1, 1]
    print(poster_p_Y_ck_0, poster_p_Y_ck_1)
    return -1 if poster_p_Y_ck_0>poster_p_Y_ck_1 else 1
    
def bayes_estimation(train_x, train_y, test_x, Lambda):
    #compute the priori probability
    pY_ck_0=Lambda
    pY_ck_1=Lambda
    for i in range(0, train_y.shape[0]):
        if train_y[i]==-1:
            pY_ck_0+=1
        else:
            pY_ck_1+=1
    pY_ck_0/=train_y.shape[0]+2*Lambda
    pY_ck_1/=train_y.shape[0]+2*Lambda
    print(pY_ck_0,pY_ck_1)
    #compute the conditional probability
    pXj_ajl_Y=np.ones((train_x.shape[1], 3, 2))
    pXj_ajl_Y*=Lambda
    for i in range(0, train_x.shape[0]):
        for j in range(0, train_x.shape[1]):
            pXj_ajl_Y[j, train_x[i][j]-1, math.floor((train_y[i]+1)/2)]+=1
    pXj_ajl_Y[:, :, 0]/=pY_ck_0*(train_y.shape[0]+2*Lambda)+2*Lambda
    pXj_ajl_Y[:, :, 1]/=pY_ck_1*(train_y.shape[0]+2*Lambda)+2*Lambda
    print(pXj_ajl_Y[:, :, 0])
    print(pXj_ajl_Y[:, :, 1])
    #compute the posterior probability
    poster_p_Y_ck_0=pY_ck_0
    poster_p_Y_ck_1=pY_ck_1
    for j in range(0, train_x.shape[1]):
        poster_p_Y_ck_0*=pXj_ajl_Y[j, test_x[j]-1, 0]
        poster_p_Y_ck_1*=pXj_ajl_Y[j, test_x[j]-1, 1]
    print(poster_p_Y_ck_0, poster_p_Y_ck_1)
    return -1 if poster_p_Y_ck_0>poster_p_Y_ck_1 else 1
    
if __name__=="__main__":
    train_x=np.array([[1,1],[1,2],[1,2],[1,1],
                [1,1],[2,1],[2,2],[2,2],
                [2,3],[2,3],[3,3],[3,2],
                [3,2],[3,3],[3,3]])
    train_y=np.array([-1,-1,1,1,-1,-1,-1,1,
                      1,1,1,1,1,1,-1])
    test_x=np.array([2,1])
    predict_label=naive_bayes(train_x, train_y, test_x)
    print(predict_label)
    Lambda=1
    predict_label=bayes_estimation(train_x, train_y, test_x, Lambda)