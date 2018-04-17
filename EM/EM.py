import numpy as np

def EM(pi, p, q, train_y):
    count0=0
    for i in range(train_y.shape[0]):
        if train_y[i]==0:
            count0+=1
    count1=train_y.shape[0]-count0;
    pi_next=0
    p_next=0
    q_next=0
    while True:
        #E step:
        miu_next_0=pi*(1-p)/(pi*(1-p)+(1-pi)*(1-q))
        miu_next_1=pi*p/(pi*p+(1-pi)*q)
        print(miu_next_0,miu_next_1,pi,p,q)
        #M step:
        pi_next=(miu_next_0*count0+miu_next_1*count1)/train_y.shape[0]
        p_next=miu_next_1*count1/(miu_next_0*count0+miu_next_1*count1)
        q_next=(1-miu_next_1)*count1/((1-miu_next_0)*count0+(1-miu_next_1)*count1)
        print(pi_next,p_next,q_next)
        if (pi-pi_next)*(pi-pi_next)+(p-p_next)*(p-p_next)+(q-q_next)*(q-q_next)<10e-6:
            break
        pi=pi_next
        p=p_next
        q=q_next
        
    return pi,p,q
        
if __name__=="__main__":
    train_y=np.mat([1,1,0,1,0,0,1,0,1,1])
    train_y=train_y.T
    pi=0.4
    p=0.6
    q=0.7
    pi,p,q=EM(pi,p,q,train_y)
    print(pi,p,q)