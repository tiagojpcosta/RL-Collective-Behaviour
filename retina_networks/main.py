#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:03:45 2017

@author: titan2
"""

import os
import numpy as np
import Network as net
from Environment3D_simple_visual_simple_movement import Environment
from Agents_para import Agents, run_parallel, run_episode
import multiprocessing as mp
import timeit
import sys

def saver(tensor, epoch,directory,n_layers):
    fnames = ['W1','b1','W2','b2','W3','b3','W4','b4','W5','b5','W6','b6']
    for i in range(int(2*n_layers)):
        fname = fnames[i] + '_epoch_' + str(epoch) + '.txt'  
        np.savetxt(os.path.join(directory, fname), tensor[i])

def load_tensor_from_files(epoch, n_layers):
    fnames = ['W1','b1','W2','b2','W3','b3']
    tensor = [None]*2*n_layers
    for i in range(int(2*n_layers)):
        fname = fnames[i] + '_epoch_' + str(epoch) + '.txt'  
        tensor[i] = np.loadtxt(fname)
    return tensor

if __name__ == "__main__":     

    
    print('finished imports')
    
    directory = './sim/'
    network_params= [100,30,6]
    lrate         = 0.07
    sigma         = 0.025
    t_steps       = 800
    anneling_rate = 0.9993
    min_anneling  = 0.002
    num_rand_pert = 33
    epochs        = 10001
    reward_vector = []
    std_reward    = []
    n_layers         = len(network_params)
    

    env_params = {'num_agents'                      : 35, 
                 'min_allowed_distance'             : 0.2,  
                 'rod_size'                         : 5,
                 'penalty_for_being_close'          : 20, 
                 'alpha_angular'                    : 1, 
                 'beta_angular'                     : 40,
                 'reward_weights'                   : np.array([1.5,1,0.5]),
                 'delta'                            : 1/8*np.pi,
                 'std_depth'                        : 0.04,    
                 'min_velocity'                     : 0.05,
                 'std_straight'                     : 0.001,
                 'type_behaviour'                   : 'sphere',
                 'mill_scl_r'                       : 2,
                 'mill_scl_xy'                      : 0.6,
                 'mill_scl_c'                       : -30,
                 }

    agents = Agents(env_params, network_params, lrate, sigma,
                 num_rand_pert, t_steps, anneling_rate, min_anneling)
    print(agents.env.type_behaviour)
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = open(os.path.join(directory,'sim_data.txt'),'w')

    f.write('Network topology: ' + str(agents.env.state_dim)+'x'
                +str(network_params[0])+'x'+str(network_params[1])+'x'+str(network_params[2])+"\n")
    f.write("\n")
    
    
    f.write("Simulation parameters:"+"\n")
    
    f.write("Sigma = "+str(sigma) + ", learning rate = " + str(lrate) + ", anneling rate = " + str(anneling_rate))
    
    f.write(", Number of random perturbations = "+str(num_rand_pert)+", time steps of simulation = " + str(t_steps) + ", Number of epochs = "+str(epochs)+"\n")
    f.write("\n")
    
    f.write("Environment parameters:"+"\n")
    
    f.write("Number of agents = " + str(env_params['num_agents']) + ", Minimum distance allowed = " + str( env_params['min_allowed_distance']))
    f.write(", Rod size = " + str(env_params['rod_size']) + ", penalty for being close = " + str(env_params['penalty_for_being_close'])+
            ", alpha angular = " + str(env_params['alpha_angular']))
    f.write(", beta angular = " + str(env_params['beta_angular']) + ", reward_weights = " + str(env_params['reward_weights'])+
            ", Delta rods = " +  str(env_params['delta']) + ", std_depth = " +  str(env_params['std_depth']))
    f.write(", minimum velocity = " +  str(env_params['min_velocity']) + ", std_straight = " +  str(env_params['std_straight']))
    f.write(", type_behaviour = " +  str(env_params['type_behaviour']) + ", mill_scl_r = " +  str(env_params['mill_scl_r']))
    f.write(", mill_scl_xy = " +  str(env_params['mill_scl_xy']) + ", mill_scl_c = " +  str(env_params['mill_scl_c']))

    f.close()


    print("Start training the agents with a network "
          +str(network_params[0])+'x'+str(network_params[1])+'x'+str(network_params[2]))
    

    print("Sigma = "+str(sigma) + ", lrate = " + str(lrate) +", num random perturbation = "
          +str(num_rand_pert)+" timesteps of sim: " + str(t_steps))
    print("N rods: " + str(agents.env.state_dim/2) + " N agents: " + str(env_params['num_agents']))
    
    start = timeit.default_timer()
    
    for epoch in range(epochs):
        
        print("Start epoch: " + str(epoch) + " with learning rate: " + str(agents.lrate))
        
        start = timeit.default_timer()
        agents.run_epoch()
        stop = timeit.default_timer()

        print("Mean reward: " + str(np.mean(agents.rewards)) + ", std reward: " + str(np.std(agents.rewards)))
        print("Duration of epoch: " + str(stop-start))
        print("")
        
        agents.update_tensor()
        reward_vector.append(np.mean(agents.rewards))
        std_reward.append(np.std(agents.rewards))
        if (epoch % 25) == 0:
            saver(agents.tensor,epoch,directory,n_layers)
            reward_file_name = os.path.join(directory,'rewards_' + str(epoch) + '.txt')
            np.savetxt(reward_file_name, np.array(reward_vector))
            std_reward_file_name = os.path.join(directory,'std_rewards_' + str(epoch) + '.txt')
            np.savetxt(std_reward_file_name, np.array(std_reward))
        
        
        
        
