import numpy as np

def selectJrand(i, m):
    j=i
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def smoSimple(train_x, train_y, C, toler, maxIter):
    dataMatrix=np.mat(train_x)
    labelMat=np.mat(train_y).transpose()
    b=0
    m,n=dataMatrix.shape
    alphas=np.zeros((m,1))
    alphas=np.mat(alphas)
    iter=0
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(np.multiply(alphas, labelMat).T* \
                      dataMatrix*dataMatrix[i,:].T)+b
            Ei=fXi-float(labelMat[i])
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or \
            ((labelMat[i]*Ei > toler) and (alphas[i]>0)):
                j=selectJrand(i,m)
                fXj=float(np.multiply(alphas, labelMat).T* \
                         (dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L==H");
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T- \
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0");
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<10e-5):
                    print("J is not moving enough")
                    continue
                alphas[i]+= labelMat[j]*labelMat[i]* \
                            (alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)* \
                    dataMatrix[i,:]*dataMatrix[i,:].T- \
                    labelMat[j]*(alphas[j]-alphaJold)* \
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)* \
                    dataMatrix[i,:]*dataMatrix[j,:].T- \
                    labelMat[j]*(alphas[j]-alphaJold)* \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                print(b1,b2)
                if (0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if(alphaPairsChanged==0):
            iter+=1
        else:
            iter=0
        print("ieration number: %d"% iter)
    return b, alphas
    
if __name__=="__main__":
    train_x=np.array([[3, 3],
                    [4, 3],
                    [1, 1]])
    train_y=np.array([1, 1, -1])
    b,alphas=smoSimple(train_x, train_y, 0.5, 0.001, 20)
    print(alphas, b)