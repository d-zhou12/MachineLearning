import numpy as np
import math

class node:
    def __init__(self, data, direction):
        self.data=data
        self.direction=direction
        self.left=None
        self.right=None
        
    def details(self):
        print(self.data, self.direction)
        if self.left!=None:
            self.left.details()
        if self.right!=None:
            self.right.details()

def kdTree(train_x, start, end, direction):
    #print(train_x, start, end)
    if start>=end or start>=train_x.shape[0]:
        return None
    if start==end-1:
        ret=node(train_x[start, :], direction)
        return ret
    train_x[start:end, :]=train_x[train_x[start:end, direction].argsort()+start]
    kdim=train_x.shape[1]
    mid=math.floor((start+end-1)/2)
    ret=node(train_x[mid], direction)
    direction=(direction+1)%kdim
    left=kdTree(train_x, start, mid, direction)
    right=kdTree(train_x, mid+1, end, direction)
    ret.left=left
    ret.right=right
    return ret
    
if __name__=="__main__":
    train_x=np.array([[2,3],
                    [5,4],
                    [9,6],
                    [4,7],
                    [8,1],
                    [7,2]])
    ret=kdTree(train_x, 0, train_x.shape[0], 0)
    ret.details()