import numpy as np

def G(cur_index, sign, index):
    if cur_index<index:
        return sign
    else:
        return -sign

def computeError(train_x, train_y, index, w):
    count1=0
    count2=0
    length=train_x.shape[0]
    for i in range(0, index):
        if train_y[i]==-1:
            count1+=w[i]
        else:
            count2+=w[i]
    for i in range(index,length):
        if train_y[i]==1:
            count1+=w[i]
        else:
            count2+=w[i]
    if(count1<count2):
        sign=1
    else:
        sign=-1
    return min(count1,count2),sign

def adaBoost(train_x, train_y, maxIter):
    f=np.zeros((maxIter,3))
    length=train_x.shape[0]
    errorCount=length
    D1=np.ones((length,1))
    w1=D1/length
    m=0
    while errorCount>0:
        print("D"+str(m),w1.T)
        minError=1.0
        index=-1
        sign=0
        for i in range(length):
            error,sign_temp=computeError(train_x, train_y, i, w1)
            if error<minError:
                minError=error
                index=i
                sign=sign_temp
        alpha1=0.5*np.log((1-minError)/minError)
        print(error,sign,index,alpha1)
        f[m][0]=alpha1
        f[m][1]=sign
        f[m][2]=index
        p_next=np.zeros((length,1))
        p_next=np.mat(p_next)
        for i in range(length):
            p_next[i]+=w1[i]*np.exp(-alpha1*train_y[i]*G(i,sign,index))
        Z1=np.sum(p_next)
        w1=p_next/Z1
        errorCount=0
        m+=1
        for i in range(0,length):
            cur=0
            for j in range(0,m):
                cur_sign=0
                if i<f[j][2]:
                    cur_sign=f[j][1]
                else:
                    cur_sign=-f[j][1]
                cur+=f[j][0]*cur_sign
            if cur*train_y[i]<=0:
                errorCount+=1
        print("errorCount numbers:", errorCount)
    return f
        
            
    

if __name__=="__main__":
    train_x=np.mat([0,1,2,3,4,5,6,7,8,9])
    train_y=np.mat([1,1,1,-1,-1,-1,1,1,1,-1])
    train_x=train_x.T
    train_y=train_y.T
    maxIter=5
    f=adaBoost(train_x,train_y,maxIter)
    print(f)