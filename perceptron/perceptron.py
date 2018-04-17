#eg 2.1-2.2 perceptron learning algorithm
import numpy as np
def original_learning(train_x,train_y):
    w0=np.zeros(train_x.shape[1])
    b0=0
    learning_rate = 1
    count = 0

    while count < train_x.shape[0]:
        for i in range(0, train_x.shape[0]):
            w_mul_x=b0
            for j in range(0, train_x.shape[1]):
                w_mul_x+=w0[j]*train_x[i,j]
            if train_y[i] * w_mul_x<=0:
                for j in range(0, train_x.shape[1]):
                    w0[j]+=learning_rate * train_y[i] * train_x[i, j]
                b0 += learning_rate * train_y[i]
                count = 0
                print(w0, b0)
            else:
                count += 1
    return w0,b0
    
def dual_learning(train_x,train_y):
    alpha=np.zeros(train_x.shape[0])
    w0=np.zeros(train_x.shape[1])
    b0 = 0
    learning_rate = 1
    count = 0
    
    gram=np.zeros((train_x.shape[0], train_x.shape[0]))
    for i in range(0, train_x.shape[0]):
        for j in range(0, train_x.shape[0]):
            gram[i,j]=0
            for k in range(0, train_x.shape[1]):
                gram[i, j]+=train_x[i, k]*train_x[j, k]
    print(gram)
    
    while count < train_x.shape[0]:
        for i in range(0,train_x.shape[0]):
            dual_value = 0
            for j in range(0, train_x.shape[0]):
                dual_value += alpha[j] * train_y[j] * gram[j, i]
            dual_value += b0
            dual_value *= train_y[i]
            if dual_value <= 0:
                alpha[i] += learning_rate
                b0 += learning_rate * train_y[i]
                count = 0
                print(alpha, b0)
            else:
                count += 1
    
    for i in range(0, train_x.shape[0]):
        for j in range(0, train_x.shape[1]):
            w0[j] += alpha[i] * train_y[i] * train_x[i, j]
            #b0 += alpha[i] * y[i]
    return w0,b0
    
if __name__=="__main__":
    train_x=np.array([[3, 3],
                    [4, 3],
                    [1, 1]])
    train_y=np.array([1, 1, -1])
    original_learning(train_x,train_y)
    w0,b0=dual_learning(train_x,train_y)
    print(w0,b0)