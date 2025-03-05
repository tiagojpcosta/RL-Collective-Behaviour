#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:56:49 2019

@author: tiago
"""

# -*- coding: utf-8 -*-
"""
22-07-19
New version symetries in xy and z
Deterministic attention
Softmax attention
"""

#implement simetric policy

import numpy as np

def sigmoid_array(x):                                        
    return 1 / (1 + np.exp(-np.clip(x,-15,15)))


class Network_rods:
    def __init__(self, tensor, stochastic_policy = False):

        self.stochastic_policy = stochastic_policy
        self.eps = np.finfo(np.float32).eps
        self.nact = int(tensor[-1].shape[0]/2)
        
        self.n_layers = int(len(tensor)/2)
        self.tensor = [[] for i in range(2*self.n_layers)]
        for i in range(2*self.n_layers):
            self.tensor[i] = np.array(tensor[i])

        
    def act(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers-1):
            x = np.maximum(np.matmul(x,self.tensor[2*i])+self.tensor[2*i+1],0) 
        i += 1
        x = sigmoid_array(np.matmul(x,self.tensor[2*i])+self.tensor[2*i+1]) 
        if self.stochastic_policy:
            return(np.clip(x[:,0:self.nact] + (0.25 * x[:,self.nact:] + 0.05) * np.random.randn(state.shape[0],self.nact),0,1))
        else:
            return x
      
    
    def update_net(self,update_tensor):
        for i in range(2*self.n_layers):
            self.tensor[i] = np.array(update_tensor[i])
        
class Network_fran:
    def __init__(self,tensor_policy, tensor_attention, stochastic_policy = False, stochastic_attention = False, symetric_policy = False):

        #support params
        self.eps = np.finfo(np.float32).eps
        self.nact = int(tensor_policy[-1].shape[0]/2)
        
        self.stochastic_policy    = stochastic_policy
        self.stochastic_attention = stochastic_attention
        self.symetric_policy      = symetric_policy
        #create policy net
        self.n_layers_policy = int(len(tensor_policy)/2)
        self.tensor_policy = [[] for i in range(2*self.n_layers_policy)]
        for i in range(2*self.n_layers_policy):
            self.tensor_policy[i] = np.array(tensor_policy[i])
            
        #create attention net
        self.n_layers_attention = int(len(tensor_attention)/2)
        self.tensor_attention = [[] for i in range(2*self.n_layers_attention)]
        for i in range(2*self.n_layers_attention):
            self.tensor_attention[i] = np.array(tensor_attention[i])
        
    def get_action(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers_policy-1):
            x = np.maximum(np.matmul(x,self.tensor_policy[2*i])+self.tensor_policy[2*i+1],0) 
        i += 1
        x = sigmoid_array(np.matmul(x,self.tensor_policy[2*i])+self.tensor_policy[2*i+1]) 
        if self.stochastic_policy:
            return(np.clip(x[:,0:self.nact] + (0.48 * x[:,self.nact:] + 0.02) * np.random.randn(state.shape[0],self.nact),0,1))
        else:
            return x
        
    def get_attention(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers_attention-1):
            x = np.maximum(np.matmul(x,self.tensor_attention[2*i])+self.tensor_attention[2*i+1],0) 
        i += 1
        x = np.exp(np.clip(np.matmul(x,self.tensor_attention[2*i])+self.tensor_attention[2*i+1],-15,15)) 
        return x/np.sum(x) 
        
  
    def act(self,state):
        a = np.zeros((len(state),3))
        if not self.symetric_policy:
            for i in range(len(state)):
                a[i,:] = np.matmul(self.get_action(state[i]).T, self.get_attention(state[i])).T           
        else:
            for i in range(len(state)):
                F_lu          = self.get_action(state[i][0])
                F_ru          = self.get_action(state[i][1])
                F_ld          = self.get_action(state[i][2])
                F_rd          = self.get_action(state[i][3])
                
                #apply symetry in space [0,1]
                F_ru[:,1] = 1 - F_ru[:,1]           
                F_ld[:,2] = 1 - F_ld[:,2]           
                F_rd[:,1] = 1 - F_rd[:,1]           
                F_rd[:,2] = 1 - F_rd[:,2]          
                
                
                G_lu          = self.get_attention(state[i][0])
                G_ru          = self.get_attention(state[i][1])
                G_ld          = self.get_attention(state[i][2])
                G_rd          = self.get_attention(state[i][3])


                a[i,:] = np.matmul((F_lu + F_ru + F_ld + F_rd).T/4,
                                   (G_lu + G_ru + G_ld + G_rd)/4).T
        return a
    
    def update_net(self,update_tensor_policy, update_tensor_attention):
        for i in range(2*self.n_layers_policy):
            self.tensor_policy[i] = np.array(update_tensor_policy[i])
            
        for i in range(2*self.n_layers_attention):
            self.tensor_attention[i] = np.array(update_tensor_attention[i])
            
    def get_action_plot(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers_policy-1):
            x = np.maximum(np.matmul(x,self.tensor_policy[2*i])+self.tensor_policy[2*i+1],0) 
        i += 1
        x = sigmoid_array(np.matmul(x,self.tensor_policy[2*i])+self.tensor_policy[2*i+1]) 
        return x[:,0:self.nact],0.48 * x[:,self.nact:] + 0.02
    
    def get_attention_plot(self,state):
        x=np.array(state,dtype = 'float32')
        for i in range(self.n_layers_attention-1):
            x = np.maximum(np.matmul(x,self.tensor_attention[2*i])+self.tensor_attention[2*i+1],0) 
        i += 1
        x = sigmoid_array(np.matmul(x,self.tensor_attention[2*i])+self.tensor_attention[2*i+1]) 
        return x[:,0],(0.5 * x[:,1])
    
    
    
if __name__ == "__main__":   
    import timeit
    np.random.seed(3)
    type_network = 'fran_model'	  
    stochastic_policy_     = True
    stochastic_attention_ = True
    symetric_policy_      = True
    net_params_policy    = [5]
    net_params_attention = [3,4,5]
    if type_network == 'fran_model':    
        net_params_policy[0:0] = [2]
        net_params_attention[0:0] = [2]
    else:
        net_params_policy[0:0] = [16]

    
    if stochastic_policy_:
        net_params_policy.append(6)
    else:
        net_params_policy.append(3)
        
    if stochastic_attention_:
        net_params_attention.append(2)
    else:
        net_params_attention.append(1)
    tensor_attention     = [[] for i in range((len(net_params_attention)-1)*2)]
    
    tensor_policy     = [[] for i in range((len(net_params_policy)-1)*2)]
    for i in range(len(net_params_policy)-1):
        tensor_policy[2*i]   = np.random.rand(net_params_policy[i],net_params_policy[i+1])-0.5
        tensor_policy[2*i+1] = np.random.rand(net_params_policy[i+1])-0.5
#        print(tensor_policy[2*i])
#        print("")
#        print(tensor_policy[2*i+1])
#        print("")
#        
#    print("")
    if type_network == 'fran_model':
        for i in range(len(net_params_attention)-1):
            tensor_attention[2*i]   = np.random.rand(net_params_attention[i],net_params_attention[i+1])
            tensor_attention[2*i+1] = np.random.rand(net_params_attention[i+1])
#        print(tensor_attention[2*i])
#        print("")
#        print(tensor_attention[2*i+1])
#        print("")
        

        net=Network_fran(tensor_policy,tensor_attention,stochastic_policy_, stochastic_attention_, symetric_policy_)
    else:
        net=Network_rods(tensor_policy,stochastic_policy = stochastic_policy_)
#    state = (2*np.ones((10,2))*np.arange(0,10)[:,None]-9)/10
    n_agents = 10
    
    if type_network == 'fran_model':    
        size_state = np.random.randint(2,6,(n_agents))
#    size_state = np.array([1, 4, 1, 4, 2, 1, 0, 1, 0, 4, 0])
        if not symetric_policy_:
            state = [2*np.random.rand(size_state[i],2) for i in range(n_agents)]
        else:
            state = [2*np.random.rand(2,size_state[i],2) for i in range(n_agents)]
#
#            temp_state = [2*np.random.rand(size_state[i],2) for i in range(n_agents)]
#            state = [np.stack((temp_state[i],temp_state[i])) for i in range(n_agents)]

    else:
        state = np.random.rand(n_agents,net_params_policy[0])
    
#    for i in range(len(state)):
#        print(state[i].shape)
#    print('')
#    print(size_state)
#    print("")
#    print(net.get_action(state[-1]))
#    print("")
#    print(net.get_attention(state[-1]))
#    print("")
#    print(net.act(state))
#    print(net.act(state))
#    if (stochastic_policy_) | (stochastic_attention_):
#        print(net.act(state))

#    print(net.tensor[0])
#    print(net.act(state))
    a= net.act(state)
    print('shape actions: ',a.shape)
#    print(a.shape)
#    state = np.random.randn(30000,30000)
#    start = timeit.default_timer()
#    for i in range(500*22*3000):
#        state.shape[0]
#    stop = timeit.default_timer()
#    print(stop-start)
    tensor_attention     = [[] for i in range((len(net_params_attention)-1)*2)]
    
    tensor_policy     = [[] for i in range((len(net_params_policy)-1)*2)]
    for i in range(len(net_params_policy)-1):
        tensor_policy[2*i]   = np.random.rand(net_params_policy[i],net_params_policy[i+1])-0.5
        tensor_policy[2*i+1] = np.random.rand(net_params_policy[i+1])-0.5
#        print(tensor_policy[2*i])
#        print("")
#        print(tensor_policy[2*i+1])
#        print("")
#        
#    print("")
    if type_network == 'fran_model':    
        for i in range(len(net_params_attention)-1):
            tensor_attention[2*i]   = np.random.rand(net_params_attention[i],net_params_attention[i+1])
            tensor_attention[2*i+1] = np.random.rand(net_params_attention[i+1])
#            print(tensor_attention[2*i])
#            print("")
#            print(tensor_attention[2*i+1])
#            print("")
        
        net.update_net(tensor_policy,tensor_attention)
    else:
        net.update_net(tensor_policy)
#    print(net.act(state))
