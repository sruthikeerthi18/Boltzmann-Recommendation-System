# -*- coding: utf-8 -*-

#importing packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import numpy as np
import pandas as pd
from torch.autograd import Variable
#importing dataset
movies=pd.read_csv('ml-1m/movies.dat', sep='::', header=None,engine='python', encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat', sep='::', header=None,engine='python', encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat', sep='::', header=None,engine='python', encoding='latin-1')
#traing and test sets
train=pd.read_csv('ml-100k/u1.base',delimiter='\t')
test=pd.read_csv('ml-100k/u1.test',delimiter='\t')
train=np.array(train,dtype=int)
test=np.array(test,dtype=int)
#total number of movies and users
nb_users = int(max(max(train[:,0]), max(test[:,0])))
nb_movies = int(max(max(train[:,1]), max(test[:,1])))
#converting the train and test sets to matrices
def model(data):
    matrix=[]
    for id_users in range(1,nb_users + 1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        rate=np.zeros(nb_movies)
        rate[id_movies-1]=id_ratings
        matrix.append(list(rate))
    return matrix

train_matrix=model(train)
test_matrix=model(test)
#converting into tensors using pytorch
#type(train_matrix)
train_tensor=torch.FloatTensor(train_matrix)
test_tensor=torch.FloatTensor(test_matrix)
#converting the matrix to binary (-1 = not rated,0 = dislike,1 = like)
train_tensor[train_tensor == 0] = -1
train_tensor[train_tensor == 1] = 0
train_tensor[train_tensor == 2] = 0
train_tensor[train_tensor >= 3] = 1
test_tensor[test_tensor == 0] = -1
test_tensor[test_tensor == 1] = 0
test_tensor[test_tensor == 2] = 0
test_tensor[test_tensor >= 3] = 1
#rbm class to define the model for the rbm

class RBM():
    def __init__(self,v,h):
        #v= no of visible nodes
        #h=no of hidden nodes
        #W= weight vector
        #a is bias wrt hidden nodes
        #b is bias wrt visible nodes
        self.W=torch.randn(h,v)
        self.a=torch.randn(1,h)
        self.b=torch.randn(1,v)
    def sample_h(self,x):
        '''sampling the probabilities P(h|v)==> sigmoid activation function 
        Gibbs sampling is used for computing the log likelihood gradient.'''
        wx=torch.mm(x,self.W.t())
        act=wx+self.a.expand_as(x)
        pHV=torch.sigmoid(act)
        val=torch.bernoulli(pHV)
        return pHV,val
    def sample_v(self,y):
        #probability that the given node is the probability the the node is activated
        wx=torch.mm(y,self.W)
        act=wx+self.b.expand_as(y)
        pVH=torch.sigmoid(act)
        val=torch.bernoulli(pVH)
        return pVH,val
    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk))
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        
nv=len(train_tensor[0])
nh=len(train_tensor[0])
rbm=RBM(nv,nh)
batch_size=150

nb_epoch = 5
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = train_tensor[id_user:id_user+batch_size]
        v0 = train_tensor[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
#vk = train_tensor[0:batch_size]
#vk[vk>=0]
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = train_tensor[id_user:id_user+1]
    vt = test_tensor[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))