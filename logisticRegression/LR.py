import numpy as np
import math

def LR(train_x, train_y, learning_rate):
    new_train_x=np.zeros((train_x.shape[0],train_x.shape[1]+1))
    for i in range(0,train_x.shape[0]):
        for j in range(0,train_x.shape[1]):
            new_train_x[i][j]=train_x[i][j]
        new_train_x[i][-1]=1
    w=np.ones(train_x.shape[1]+1)
    #loss=compute_loss(new_train_x, train_y, w)
    error=np.ones(train_x.shape[1]+1)*10
    delta=0.99
    while(np.linalg.norm(error,1)>10e-6):
        temp=np.copy(w)
        for i in range(0, new_train_x.shape[0]):
            wx=np.dot(w,new_train_x[i])
            if (1/(1+math.exp(wx))<=delta and train_y[i]==0) or \
            (math.exp(wx)/(1+math.exp(wx))<=delta and train_y[i]==1):
                count=0
                w+=learning_rate*train_y[i]*new_train_x[i]-learning_rate* \
                new_train_x[i]*math.exp(wx)/(1+math.exp(wx))
        error=temp-w
        loss=compute_loss(new_train_x, train_y, w)
        #print(loss)
    scale=1/np.linalg.norm(w,2)
    return w*scale
        
def compute_loss(train_x, train_y, w):
    logloss=0.0
    for i in range(0,train_x.shape[0]):
        wx=np.dot(w,train_x[i])
        if wx!=0:
            logloss-=train_y[i]*wx-math.log(1+math.exp(wx))
    return logloss
        
if __name__=="__main__":
    train_x=np.array([[3, 3],
                    [4, 3],
                    [1, 1]])
    train_y=np.array([1, 1, 0])
    w=LR(train_x, train_y, 0.01)
    print(w)