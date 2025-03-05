#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:25:10 2017

@author: root
"""
import Network as net
import numpy as np
from Environment3D_simple_visual_simple_movement import Environment
import timeit
import multiprocessing as mp


class Agents:
    def __init__(self, env_params, network_params, lrate, sigma,
                 num_rand_pert, t_steps, anneling_rate, min_anneling = 0.1):
        
        #Create environment
        self.env  = Environment(**env_params)

        self.n_layers         = len(network_params)
        self.network_param    = np.zeros(self.n_layers + 1).astype('int32')
        self.network_param[0] = self.env.state_dim
        self.network_param[1:] = network_params
   
        #generalize place of ori input

        self.tensor        = [None] * 2 * self.n_layers
        self.inicialize_tensor()          
        self.net = net.Network(self.tensor)
        
        #Training params
        self.lrate         = lrate
        self.sigma         = sigma
        self.t_steps       = t_steps
#        self.N_layers      = N_layers
        self.min_anneling  = min_anneling
        self.num_rand_pert = num_rand_pert
        
        #Training Variables
        self.anneling_rate = anneling_rate
        self.epsilon       = [None] * 2 * self.n_layers
        self.grad          = [None] * 2 * self.n_layers
        self.rewards       = np.zeros(2 * self.num_rand_pert).astype('float32')   
                               
        #assistance variables
        self.num_cores = mp.cpu_count()-1
        
    def create_random_pert(self):
        # Create n Random perturbations on the parameters of the NN drawn from a normal distribution of mean 
#         Mean 0 and std self.sigma
        for i in range(self.n_layers):
                self.epsilon[2*i]   = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_param[i],self.network_param[i+1]),).astype('float32')
                self.epsilon[2*i+1] = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_param[i+1],),).astype('float32')
                        
    def compute_gradient(self):
        
        #normalize rewards
        norm_rewards = (self.rewards-np.mean(self.rewards))/np.std(self.rewards)
        
        #Estimate gradient by ES
        for i in range(2 * self.n_layers):
            self.grad[i] = np.matmul(self.epsilon[i].T,norm_rewards[0:self.num_rand_pert]) + np.matmul(-self.epsilon[i].T,norm_rewards[self.num_rand_pert:])
        
    def update_tensor(self):
        
#        self.anneling()
        self.compute_gradient()
        update_constant = self.lrate/(2*self.num_rand_pert*self.sigma)
        for i in range(2 * self.n_layers):
            self.tensor[i] = self.tensor[i] + update_constant*self.grad[i].T

        
    def inicialize_tensor(self):
        for i in range(self.n_layers - 1):
            self.tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_param[i])/2.0,[self.network_param[i] ,self.network_param[i+1]]).astype('float32')
            self.tensor[2*i+1]  = np.zeros([self.network_param[i+1]]).astype('float32')
        i += 1   
        self.tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_param[i])/2.0,[self.network_param[i],self.network_param[i+1]]).astype('float32')
        self.tensor[2*i+1]  = -0.5*np.ones([self.network_param[i+1]]).astype('float32')

        
    def load_tensor(self,tensor):
        for i in range(2 * self.n_layers):
            self.tensor[i] = tensor[i] 

    def update_lrate(self):
        self.lrate = np.maximum(self.anneling_rate*self.lrate,self.min_anneling)
     
    def run_epoch(self):
        #Create random perturbations
        self.env.reset(0)      #i==0, Hard reset

        self.create_random_pert()
        #Run simulation and collect rewards
        results=np.array(run_parallel(self.net, self.env, self.tensor, 
                                            self.epsilon, self.num_rand_pert, 
                                            self.n_layers, self.t_steps, self.num_cores))
        self.rewards = np.concatenate((results[:,0],results[:,2]))
#        print(np.concatenate((results[:,1],results[:,3])))
#        print(self.rewards)
        #update learning rate and tensor
        self.update_tensor()
        self.update_lrate()    

#    def get_trajectories_sim(self):
#        print("Creating a plot of the simulation")
#        
#        self.net.update_net(self.tensor) 
#        
#        state = self.env.reset(0)        # zero means an hard reset, one means soft reset
#
#        reward_simulation = 0
#        trajectories = np.zeros((3,self.env.num_agents,self.t_steps+1))
#      
#        trajectories[:,:,0] = self.env.positions.T
#        for t in range(self.t_steps):
#            a = self.net.act(state)
#            state, reward, _, d = self.env.step(a)
#            reward_simulation += reward
#            trajectories[:,:,t+1] = self.env.positions.T
#            if d == True:
#                print("WE FUCKED UP")
#                break
# 
#        print("Accumulative reward = " + str(reward_simulation))
#        return trajectories  
    
    
def run_parallel(net, env, tensor, epsilon, num_rand_pert, n_layers, t_steps, num_cores):
#        Run Epoch in parallel
    pool = mp.Pool(num_cores)
   
    list_arguments = [(net, env, tensor, [epsilon[j][i] for j in range(2*n_layers)], t_steps,) for i in range(num_rand_pert)]
    results = pool.starmap(run_episode,list_arguments)  
    pool.close()
    return results
  
def run_episode(net, env, tensor, pert, t_steps):
    # Run the episode     
    tensor_pert = [tensor[i] + pert[i] for i in range(len(tensor))]
    net.update_net(tensor_pert)
    accumulative_reward_positive_pert = 0
                
    state = env.reset(0)   #i== 1, Soft reset, i==0 Hard reset
    for t in range(t_steps):
        a = net.act(state)
        state, reward, _, = env.step(a)
        accumulative_reward_positive_pert += reward
          
    tensor_pert = [tensor[i] - pert[i] for i in range(len(tensor))]
    net.update_net(tensor_pert)
    accumulative_reward_negative_pert = 0
                
    state = env.reset(1)   #i== 1, Soft reset, i==0 Hard reset
    for t in range(t_steps):
        a = net.act(state)
        state, reward, _ = env.step(a)
        accumulative_reward_negative_pert += reward
        
    return np.array([accumulative_reward_positive_pert,1,accumulative_reward_negative_pert,0])
    

 

    
    
if __name__ == "__main__":     

    
    print('finished imports')

    network_params= [30,10,6]

    lrate         = 0.5
    sigma         = 0.1
    t_steps       = 500
    anneling_rate = 0.995
    num_rand_pert = 3
    epochs        = 5
    reward_vector = []
    env_params = {'num_agents'                      : 30, 
                  'min_allowed_distance'             : 0.2,  
                  'rod_size'                         : 5,
                  'penalty_for_being_close'          : 20, 
                  'min_angular_displacement'         : np.pi/32, 
                  'scalling_constant_a_displacement' : 10,
                  'reward_weights'                   : np.array([1,1]),
                  'delta'                            : 1/16*np.pi
                 }

    
    agents = Agents(env_params, network_params, lrate, sigma,
                 num_rand_pert, t_steps, anneling_rate, min_anneling = 0.1)
    
#    print("Start training the agents with a network "
#          +str(N1)+"x"+str(N2)+"x"+str(Nact)+" and sigma = "+str(sigma))
    start = timeit.default_timer()
#    agents.create_random_pert()
#    a = run_parallel(agents.net, agents.env, agents.tensor, 
#                                            agents.epsilon, agents.num_rand_pert, 
#                                            agents.n_layers, agents.t_steps, agents.num_cores) 
    stop = timeit.default_timer()
    print(stop-start)
#    print(agents.env.reset(0))

    
    for i in range(epochs):
        print(i)
        agents.run_epoch()
        print(np.mean(agents.rewards))
        stop = timeit.default_timer()
        print(stop-start)
        
        
        
        
#    print("Start training the agents with a network "
#          +str(N1)+"x"+str(N2)+"x"+str(Nact)+" and sigma = "+str(sigma))
#    start = timeit.default_timer()
#    agents.run_epoch()
#    stop = timeit.default_timer()
#    print(stop-start)

#5 epochs
#serie - 58.63083334579297
#parallel - 41.11404255070861
       
#1 epoch 
#serie - 11.823284480598879/41.371547235837674
# parallel - 8/

    
    
    
    
