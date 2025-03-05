# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:54:40 2018

@author: Tiago Costa
"""

import numpy as np

def sigmoid_array(x):                                        
    return 1 / (1 + np.exp(-np.clip(x,-15,15)))

class Network:
    def __init__(self, tensor):

        self.n_layers = int(len(tensor)/2)
        self.tensor = [[] for i in range(2*self.n_layers)]
        for i in range(2*self.n_layers):
            self.tensor[i] = np.array(tensor[i])
        self.eps = np.finfo(np.float32).eps
        self.nact = int(tensor[-1].shape[0]/2)
        
    def act(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers-1):
            x = np.maximum(np.matmul(x,self.tensor[2*i])+self.tensor[2*i+1],0) 
        i += 1
        x = sigmoid_array(np.matmul(x,self.tensor[2*i])+self.tensor[2*i+1]) 
        return(np.clip(x[:,0:self.nact] + (0.25 * x[:,self.nact:] + 0.05) * np.random.randn(state.shape[0],self.nact),0,1))
      
    
    def update_net(self,update_tensor):
        for i in range(2*self.n_layers):
            self.tensor[i] = np.array(update_tensor[i])
        
if __name__ == "__main__":   
    import timeit
    
    print("")
    print("")
    print("newteste")
    n_samples=5
    Nin = 2; Nact = 6; N1 = 4; N2 = 5
    W1_up = np.ones(shape = (Nin,N1),dtype = np.float32)
    b1_up = -np.zeros(shape=(N1),dtype = np.float32)
    W2_up = np.ones(shape=(N1,N2),dtype = np.float32)
    b2_up = np.zeros(shape=(N2),dtype = np.float32)
    W3_up = np.ones(shape=(N2,Nact),dtype = np.float32)
    b3_up = np.zeros(shape=(Nact),dtype = np.float32)
    
    W1_up[0][0]=30
    W2_up[0][0]=0
    W3_up[0][0]=2
    
    tensor=[W1_up, b1_up, W2_up, b2_up, W3_up, b3_up]

    net=Network(tensor)
#    print(net.act(state))
    W1_up2 = np.ones(shape = (Nin,N1),dtype = np.float32)+1
    b1_up2 = -np.zeros(shape=(N1),dtype = np.float32)+1
    W2_up2 = np.ones(shape=(N1,N2),dtype = np.float32)+1
    b2_up2 = np.zeros(shape=(N2),dtype = np.float32)+1
    W3_up2 = np.ones(shape=(N2,Nact),dtype = np.float32)+1
    b3_up2 = np.zeros(shape=(Nact),dtype = np.float32)+1
    update_tensor = [W1_up2, b1_up2, W2_up2, b2_up2, W3_up2, b3_up2]
    print(tensor)
    print("")
    net.update_net(update_tensor)
    print(tensor)
    print("")
    for i in range(len(tensor)):
        print(tensor[i])
        print(net.tensor[i])
        print("")
#    print(net.tensor[0])
#    print(net.act(state))
#    a= net.act(state)
#    state = np.random.randn(30000,30000)
#    start = timeit.default_timer()
#    for i in range(500*22*3000):
#        state.shape[0]
#    stop = timeit.default_timer()
#    print(stop-start)
