import numpy as np
import math

age_dict={0:"青年",1:"中年",2:"老年"}
work_dict={0:"否",1:"是"}
house_dict={0:"否",1:"是"}
evaluate_dict={0:"一般",1:"好",2:"非常好"}
class_dict={0:"否",1:"是"}
features=[age_dict,work_dict,house_dict,evaluate_dict]

def print_origin(train_x, train_y):
    print("年龄 有工作 有自己的房子 信贷情况 类别")
    for i in range(0, train_x.shape[0]):
        print(age_dict[train_x[i][0]],work_dict[train_x[i][1]],
              house_dict[train_x[i][2]],evaluate_dict[train_x[i][3]],
              class_dict[train_y[i]])

def info_gain(train_x,train_y,A,visited_A):
    #A is features set
    #return gain information for ID3
    #and gain information ratio for C4.5
    g_DA=np.zeros(train_x.shape[1])
    C_k=np.zeros(len(class_dict))
    H_D=0
    for i in range(0, train_y.shape[0]):
        C_k[train_y[i]]+=1
    p_C_k=C_k/train_y.shape[0]
    for i in range(0,len(class_dict)):
        if p_C_k[i]>0:#pay attention to the situation of p_C_k[i]==0!!!
            H_D-=p_C_k[i]*math.log(p_C_k[i],2)
    for x in range(0, train_x.shape[1]):
        H_D_given_A=0
        for i in range(0, len(A[x])):
            if visited_A[x] is False:
                Di=0
                Dik=np.zeros(len(class_dict))
                for j in range(0, train_x.shape[0]):
                    if train_x[j][x]==i:
                        Di+=1
                        Dik[train_y[j]]+=1
                p_Di=Di/train_y.shape[0]
                p_Dik_given_Di=Dik/Di
                H_Di=0
                for k in range(0, len(class_dict)):
                    if p_Dik_given_Di[k]>0:#pay attention to the situation of Dik==0
                        H_Di-=p_Dik_given_Di[k]*math.log(p_Dik_given_Di[k],2)
                H_D_given_A+=p_Di*H_Di
        g_DA[x]=H_D-H_D_given_A
    gR_DA=g_DA/H_D
    return g_DA,gR_DA

class node:
    def __init__(self, feature, isleaf, category, A):
        self.feature=feature
        self.isleaf=isleaf
        self.category=category
        if not isleaf:
            self.child=[]
            for i in range(0, len(A[feature])):
                self.child.append(None)

    def details(self):
        if not self.isleaf:
            print(self.feature,self.isleaf,self.category)
            for i in range(0,len(self.child)):
                if self.child[i] is not None:
                    self.child[i].details()
        else:
            print(class_dict[self.category])
        
        
def generate_decision_tree(train_x, train_y, A, visited_A, feature_choose_type):
    #D is train_x and train_y
    #A is the feature sets
    
    if train_y.shape[0]==0:
        return None
    #judge the inst in D is or not the same category
    flag=True
    for i in range(0,train_y.shape[0]):
        for j in range(1, train_y.shape[0]):
            if train_y[i]!=train_y[j]:
                flag=False
                break
    if flag is True:
        root=node(-1, True, train_y[0], A)
        return root
    #if A is empty, return the max count of category inst
    flag=False
    for i in visited_A:
        if i is False:
            flag=True
            break
    if flag is False:
        C_k=np.zeros(len(class_dict))
        for i in range(0, train_y.shape[0]):
            C_k[train_y[i]]+=1
        cate=-1
        max_cate=0
        for i in range(0, C_k.shape[0]):
            if C_k[i]>max_cate:
                max_cate=C_k[i]
                cate=i
        root=node(-1, True, train_y[cate], A)
        return root
    #compute the infomation gain
    gda=info_gain(train_x, train_y,A,visited_A)
    choose=gda[feature_choose_type]
    max_temp=0
    max_index=-1
    for i in range(0, choose.shape[0]):
        if max_temp<choose[i]:
            max_temp=choose[i]
            max_index=i
    print(choose,max_index)
    root=node(max_index, False, -1, A)
    visited_A[max_index]=True
    for i in range(0, len(A[max_index])):
        next_train_x=train_x
        next_train_y=train_y
        j=0
        while j<next_train_x.shape[0]:
            if next_train_x[j][max_index]!=i:
                next_train_x=np.delete(next_train_x, j, axis=0)
                next_train_y=np.delete(next_train_y, j, axis=0)
            else:
                j+=1
        root.child[i]=generate_decision_tree(next_train_x, next_train_y, 
                                             A, visited_A, feature_choose_type)
    return root
        
    
    
if __name__=="__main__":
    train_x=np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],
                      [0,0,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1],
                      [1,0,1,2],[1,0,1,2],[2,0,1,2],[2,0,1,1],
                      [2,1,0,1],[2,1,0,2],[2,0,0,0]])
    train_y=np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
    A=[age_dict,work_dict,house_dict,evaluate_dict]
    visited_A=[False, False, False, False, False]
    feature_choose_type=0
    print_origin(train_x, train_y)
    root=generate_decision_tree(train_x,train_y,A,visited_A,feature_choose_type)
    root.details()