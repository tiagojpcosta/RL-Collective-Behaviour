#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:25:10 2017

@author: root
"""

"""
22-07-19
Deterministic attention change line 59 for future implementations. Now is like 
that to prevent mistakes
"""
from Networks import Network_fran
import numpy as np
from Environment3D_simple_visual_simple_movement import Environment
import multiprocessing as mp


class Agents:
    def __init__(self, env_params, network_params_policy, network_params_attention,
                 stochastic_policy, stochastic_attention, symetric_policy, symetric_env,
                 lrate, sigma, num_rand_pert, t_steps, 
                 anneling_rate = 1, min_anneling = 0.1, 
                 sigma_anneling_rate = 1, sigma_min_anneling = 0.001):
        
        #Policy Params
        self.stochastic_policy    = stochastic_policy
        self.stochastic_attention = stochastic_attention
        self.symetric_policy      = symetric_policy  ###still not implemented
        self.symetric_env         = symetric_env
        
        #Create environment
        env_params['type_visual_system'] = 'fran_model'
        env_params['symetric_policy']    = symetric_policy


#        env_params['symetric_policy'] = 'fran_model'  ###still not implemented
        self.env  = Environment(**env_params)
        
        
        #Policy network params
        self.n_layers_policy          = len(network_params_policy) + 1
        self.network_params_policy     = np.zeros(self.n_layers_policy + 1).astype('int32')
        self.network_params_policy[0]  = self.env.state_dim
        self.network_params_policy[1:-1] = network_params_policy
        
        if self.stochastic_policy == True:
            self.network_params_policy[-1] = 6
        else:
            self.network_params_policy[-1] = 3

        #Attention network params
        self.n_layers_attention             = len(network_params_attention) + 1
        self.network_params_attention       = np.zeros(self.n_layers_attention + 1).astype('int32')
        self.network_params_attention[0]    = self.env.state_dim
        self.network_params_attention[1:-1] = network_params_attention
        
        if self.stochastic_attention == True:
            self.network_params_attention[-1] = 1
        else:
            self.network_params_attention[-1] = 1
        

        self.policy_tensor        = [None] * 2 * self.n_layers_policy
        self.attention_tensor     = [None] * 2 * self.n_layers_attention

        #initialize tensor and build network
        self.inicialize_tensors()  
        self.net = Network_fran(self.policy_tensor,self.attention_tensor,
                                self.stochastic_policy,self.stochastic_attention,symetric_policy)
        #Training params
        self.lrate         = lrate
        self.sigma         = sigma
        self.t_steps       = t_steps
        self.num_rand_pert = num_rand_pert

        self.anneling_rate       = anneling_rate
        self.min_anneling        = min_anneling
        self.sigma_anneling_rate = sigma_anneling_rate
        self.sigma_min_anneling  = sigma_min_anneling
        
        #Training Variables
        self.policy_perturbation       = [None] * 2 * self.n_layers_policy
        self.attention_perturbation    = [None] * 2 * self.n_layers_attention

        self.policy_grad               = [None] * 2 * self.n_layers_policy
        self.attention_grad            = [None] * 2 * self.n_layers_attention
        
        self.rewards                   = np.zeros(2 * self.num_rand_pert).astype('float32')   
                               
        #assistance variables
        self.num_cores = np.minimum(num_rand_pert,mp.cpu_count()-1)

        
    def inicialize_tensors(self):
        #Inicialize policy tensor with xavier inicialization
        for i in range(self.n_layers_policy - 1):
            self.policy_tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_params_policy[i])/2.0,
                                          [self.network_params_policy[i] ,self.network_params_policy[i+1]]).astype('float32')
            self.policy_tensor[2*i+1]  = np.zeros([self.network_params_policy[i+1]]).astype('float32')
            
        i += 1   
        self.policy_tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_params_policy[i])/2.0,
                                      [self.network_params_policy[i],self.network_params_policy[i+1]]).astype('float32')
        self.policy_tensor[2*i+1]  = -0.5*np.ones([self.network_params_policy[i+1]]).astype('float32')
        
        #Inicialize attention tensor with xavier inicialization
        for i in range(self.n_layers_attention - 1):
            self.attention_tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_params_attention[i])/2.0,
                                          [self.network_params_attention[i] ,self.network_params_attention[i+1]]).astype('float32')
            self.attention_tensor[2*i+1]  = np.zeros([self.network_params_attention[i+1]]).astype('float32')
            
        i += 1   
        self.attention_tensor[2*i]    = np.random.normal(0, 1/np.sqrt(self.network_params_attention[i])/2.0,
                                      [self.network_params_attention[i],self.network_params_attention[i+1]]).astype('float32')
        self.attention_tensor[2*i+1]  = -0.5*np.ones([self.network_params_attention[i+1]]).astype('float32')
        
        
    def create_random_pert(self):
        # Create n Random perturbations on the parameters of the NN drawn from a normal distribution of mean 
#         Mean 0 and std self.sigma
        for i in range(self.n_layers_policy):
            self.policy_perturbation[2*i]   = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_params_policy[i],self.network_params_policy[i+1]),).astype('float32')
            self.policy_perturbation[2*i+1] = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_params_policy[i+1],),).astype('float32')
            
        for i in range(self.n_layers_attention):
            self.attention_perturbation[2*i]   = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_params_attention[i],self.network_params_attention[i+1]),).astype('float32')
            self.attention_perturbation[2*i+1] = np.random.normal(0,self.sigma,(self.num_rand_pert,)+(self.network_params_attention[i+1],),).astype('float32')
                        
    def compute_gradient(self):
        
        #normalize rewards
        norm_rewards = (self.rewards-np.mean(self.rewards))/np.std(self.rewards)
        
        #Estimate gradient by ES
        for i in range(2 * self.n_layers_policy):
            self.policy_grad[i] = np.matmul(self.policy_perturbation[i].T,norm_rewards[0:self.num_rand_pert]) + np.matmul(-self.policy_perturbation[i].T,norm_rewards[self.num_rand_pert:])
            
        for i in range(2 * self.n_layers_attention):
            self.attention_grad[i] = np.matmul(self.attention_perturbation[i].T,norm_rewards[0:self.num_rand_pert]) + np.matmul(-self.attention_perturbation[i].T,norm_rewards[self.num_rand_pert:])
        
    def update_tensor(self):
        
#        self.anneling()
        self.compute_gradient()
        update_constant = self.lrate/(2*self.num_rand_pert*self.sigma)
        
        for i in range(2 * self.n_layers_policy):
            self.policy_tensor[i] = self.policy_tensor[i] + update_constant*self.policy_grad[i].T
            
        for i in range(2 * self.n_layers_attention):
            self.attention_tensor[i] = self.attention_tensor[i] + update_constant*self.attention_grad[i].T
            
    def update_rates(self):
        self.lrate = np.maximum(self.anneling_rate*self.lrate,self.min_anneling)
        self.sigma = np.maximum(self.sigma_anneling_rate*self.sigma,self.sigma_min_anneling)

    def run_epoch(self):
        #Create random perturbations
        self.env.reset(0)      #i==0, Hard reset

        self.create_random_pert()
        #Run simulation and collect rewards
        results=np.array(run_parallel(self.net, self.env, self.policy_tensor, self.attention_tensor,
                                            self.policy_perturbation, self.attention_perturbation,
                                            self.n_layers_policy, self.n_layers_attention,
                                            self.num_rand_pert, self.t_steps, self.num_cores,self.symetric_env ))
        self.rewards = np.concatenate((results[:,0],results[:,2]))
#        print(np.concatenate((results[:,1],results[:,3])))
#        print(self.rewards)
        #update learning rate and tensor
        self.update_tensor()
        self.update_rates()    
        
    def load_tensor(self,policy_tensor,attention_tensor):
        for i in range(2 * self.n_layers_policy):
            self.policy_tensor[i] = policy_tensor[i] 
            
        for i in range(2 * self.n_layers_attention):
            self.attention_tensor[i] = attention_tensor[i] 
            
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
    

def run_parallel(net, env, policy_tensor, attention_tensor,
                 policy_perturbation, attention_perturbation,
                 n_layers_policy, n_layers_attention,
                 num_rand_pert, t_steps, num_cores, symetric_env):
#        Run Epoch in parallel
    pool = mp.Pool(num_cores)
   
    list_arguments = [(net, env, policy_tensor, attention_tensor,
                      [policy_perturbation[j][i] for j in range(2*n_layers_policy)], [attention_perturbation[j][i] for j in range(2*n_layers_attention)], 
                      t_steps,) for i in range(num_rand_pert)]
    if symetric_env:
    	results = pool.starmap(run_episode_symetric,list_arguments)  
    else:
    	results = pool.starmap(run_episode,list_arguments)  
    pool.close()
    return results
  

def run_episode(net, env, 
                policy_tensor,attention_tensor, 
                policy_perturbation,attention_perturbation, 
                t_steps):
    # Run the episode     
    policy_tensor_pert = [policy_tensor[i] + policy_perturbation[i] for i in range(len(policy_tensor))]
    attention_tensor_pert = [attention_tensor[i] + attention_perturbation[i] for i in range(len(attention_tensor))]
    net.update_net(policy_tensor_pert, attention_tensor_pert)
    accumulative_reward_positive_pert = 0
                
    state = env.reset(0)   #i== 1, Soft reset, i==0 Hard reset
    for t in range(t_steps):
        a = net.act(state)
        state, reward, _, = env.step(a)
        accumulative_reward_positive_pert += reward
        
    policy_tensor_pert = [policy_tensor[i] - policy_perturbation[i] for i in range(len(policy_tensor))]
    attention_tensor_pert = [attention_tensor[i] - attention_perturbation[i] for i in range(len(attention_tensor))]
    net.update_net(policy_tensor_pert, attention_tensor_pert)
    accumulative_reward_negative_pert = 0
                
    state = env.reset(1)   #i== 1, Soft reset, i==0 Hard reset
    for t in range(t_steps):
        a = net.act(state)
        state, reward, _ = env.step(a)
        accumulative_reward_negative_pert += reward
        
    return np.array([accumulative_reward_positive_pert,1,accumulative_reward_negative_pert,0])
    
def run_episode_symetric(net, env, 
                policy_tensor,attention_tensor, 
                policy_perturbation,attention_perturbation, 
                t_steps):
    # Run the episode     
    policy_tensor_pert = [policy_tensor[i] + policy_perturbation[i] for i in range(len(policy_tensor))]
    attention_tensor_pert = [attention_tensor[i] + attention_perturbation[i] for i in range(len(attention_tensor))]
    net.update_net(policy_tensor_pert, attention_tensor_pert)
    accumulative_reward_positive_pert = 0
                
    state = env.reset(0)   #i== 1, Soft reset, i==0 Hard reset
    env.activate_symetry = True    
	
    for t in range(t_steps):
        a = net.act(state)
        #print('action',a)
        state, reward, _, = env.step(a)
        accumulative_reward_positive_pert += reward

    policy_tensor_pert = [policy_tensor[i] - policy_perturbation[i] for i in range(len(policy_tensor))]
    attention_tensor_pert = [attention_tensor[i] - attention_perturbation[i] for i in range(len(attention_tensor))]
    net.update_net(policy_tensor_pert, attention_tensor_pert)
    accumulative_reward_negative_pert = 0

    state = env.reset(1)   #i== 1, Soft reset, i==0 Hard reset
    env.activate_symetry = True    
    for t in range(t_steps):
        a = net.act(state)
        #print('action',a)
        state, reward, _ = env.step(a)
        accumulative_reward_negative_pert += reward
    #print(accumulative_reward_positive_pert)
    #print(accumulative_reward_negative_pert)
    return np.array([accumulative_reward_positive_pert,1,accumulative_reward_negative_pert,0])
 

    
    
if __name__ == "__main__": 
    import os    
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"
    import timeit

    print('finished imports')
    
    env_params = {'num_agents'                      : 15, 
                 'box_multiplier'                   : 0.06,
                 'min_allowed_distance'             : 0.2,  
                 'rod_size'                         : 5,
                 'penalty_for_being_close'          : 20, 
                 'alpha_angular'                    : 1, 
                 'beta_angular'                     : 30,
                 'reward_weights'                   : np.array([1.5,1,0.5]),
                 'delta'                            : 1/8*np.pi,
                 'std_depth'                        : 0.025,    
                 'min_velocity'                     : 0.05,
                 'std_straight'                     : 0.001,
                 'type_behaviour'                   : 'milling',
                 'mill_scl_r'                       : 2,
                 'mill_scl_xy'                      : 0.6,
                 'mill_scl_c'                       : -35,
                 }
    
    network_params_policy    = [30,6]
    network_params_attention = [100,30,4]
    stochastic_policy = False
    stochastic_attention= False
    symetric_policy = True
    symetric_env  = True
    
    lrate         = 0.5
    sigma         = 0.1
    t_steps       = 10
    anneling_rate = 0.995
    min_anneling  = 0.000123
    num_rand_pert = 1
    epochs        = 1

    reward_vector = []


    agents = Agents(env_params, network_params_policy, network_params_attention,
                    stochastic_policy, stochastic_attention, symetric_policy,symetric_env,
                    lrate, sigma, num_rand_pert, t_steps, 
                    anneling_rate, min_anneling = min_anneling,
                    sigma_anneling_rate = 0.5, sigma_min_anneling = 0.3)

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
        
        

#5 epochs
#serie - 58.63083334579297
#parallel - 41.11404255070861
#       
#1 epoch 
#serie - 11.823284480598879/41.371547235837674
# parallel - 8/

    
    
    
    
