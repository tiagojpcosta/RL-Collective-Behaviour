#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:03:45 2017

@author: titan2
"""

import os
import numpy as np
import timeit

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
#rebuild saver function
def saver(tensor, epoch,directory,n_layers, id_tensor = ''):
    a = ['W','b']
    for i in range(int(2*n_layers)):
        fname = id_tensor + a[i%2] +str(int(i/2))+ '_epoch_' + str(epoch) + '.txt'  
        np.savetxt(os.path.join(directory, fname), tensor[i])

#def load_tensor_from_files(epoch, n_layers):
#    fnames = ['W1','b1','W2','b2','W3','b3']
#    tensor = [None]*2*n_layers
#    for i in range(int(2*n_layers)):
#        fname = fnames[i] + '_epoch_' + str(epoch) + '.txt'  
#        tensor[i] = np.loadtxt(fname)
#    return tensor

if __name__ == "__main__":     
#    np.random.seed(3)
    
    print('finished imports')
    directory = './symetric_'

    #Choose type of visual system fran_model
    type_visual_system = 'fran_model'
#    type_visual_system = 'rods'
    
    network_params_policy    = [100,30]
    network_params_attention = [100,30]
    n_layers_policy     = len(network_params_policy) + 1
    n_layers_attention  = len(network_params_attention) + 1

    stochastic_policy = True
    stochastic_attention= False
    symetric_policy = True 
    symetric_env = True  
    
    lrate               = 0.1
    anneling_rate       = 0.9998
    min_anneling        = 0.005
    
    sigma               = 0.1
    sigma_anneling_rate = 0.9997
    sigma_min_anneling  = 0.005
    
    epochs              = 10001
    t_steps             = 800
    num_rand_pert       = 30
   

    reward_vector       = []
    std_reward          = []


    env_params = {'num_agents'                      : 35, 
                 'min_allowed_distance'             : 0.2,  
                 'box_multiplier'                   : 0.06,
                 'rod_size'                         : 5,
                 'penalty_for_being_close'          : 10, 
                 'alpha_angular'                    : 1, 
                 'beta_angular'                     : 40,
                 'reward_weights'                   : np.array([1.5,1,0.5]),
                 'delta'                            : 1/8*np.pi,
                 'std_depth'                        : 0.04,    
                 'min_velocity'                     : 0.02,
                 'std_straight'                     : 0.001,
                 'type_behaviour'                   : 'ec_milling',
                 'mill_scl_r'                       : 2,
                 'mill_scl_xy'                      : 0.6,
                 'mill_scl_c'                       : -30,
                 'topological_neighbors'            : True,
                 'num_neighbors'                    : 6,
                 }

    if type_visual_system == 'fran_model':
        from Agents_para_fran import Agents, run_parallel, run_episode
        
        agents = Agents(env_params, network_params_policy, network_params_attention,
                        stochastic_policy, stochastic_attention, symetric_policy,symetric_env,
                        lrate, sigma, num_rand_pert, t_steps, 
                        anneling_rate, min_anneling,
                        sigma_anneling_rate, sigma_min_anneling)

    else:
        from Agents_para_rods import Agents, run_parallel, run_episode

        agents = Agents(env_params, network_params_policy, stochastic_policy, lrate, sigma,
                        num_rand_pert, t_steps, anneling_rate, min_anneling,
                        sigma_anneling_rate, sigma_min_anneling)

    directory = directory +env_params['type_behaviour']+'_'+ type_visual_system + '_' + str(stochastic_policy)
    
    if type_visual_system == 'fran_model':
        directory = directory +'_' + str(stochastic_attention)
        directory = directory +'_' + str(symetric_policy)
        directory = directory +'_' + str(symetric_env)
        
    dir_exists = True
    dir_counter = -1
    while dir_exists :
        dir_counter += 1
        dir_exists = os.path.exists(directory+ ' ' + str(dir_counter))     
    directory = directory + ' ' + str(dir_counter) +'/'
    os.makedirs(directory)

    f = open(os.path.join(directory,'sim_data.txt'),'w')
    
    policy_network_info = 'policy_network_topology: ' + str(agents.env.state_dim)
    for elem in network_params_policy:
        policy_network_info = policy_network_info + 'x' + str(elem)
    f.write(policy_network_info+"\n")
    
    if type_visual_system == 'fran_model':
        attention_network_info = 'attention_network_topology: ' + str(agents.env.state_dim)
        for elem in network_params_attention:
            attention_network_info = attention_network_info + 'x' + str(elem)
        f.write(attention_network_info+"\n")
        
    f.write("stochastic_policy: " + str(stochastic_policy) +"\n")
    if type_visual_system == 'fran_model':
        f.write("stochastic_attention: " + str(stochastic_attention) +"\n")
        f.write("symetric_policy: " + str(symetric_policy) +"\n")
        f.write("symetric_env: " + str(symetric_env) +"\n")

    f.write("\n")
    
    f.write("Simulation parameters:"+"\n")
    
    f.write("type_visual_system " + type_visual_system +"\n")
    f.write("sigma = "+str(sigma) +"\n" +
            "sigma_anneling_rate = " + str(sigma_anneling_rate) +"\n" +
            "sigma_min_anneling_ = " + str(sigma_min_anneling) +"\n" +
            "lrate = " + str(lrate) + "\n" +
            "anneling rate = " + str(anneling_rate) +"\n" +
            "min_anneling = " + str(min_anneling) +"\n")

    f.write("num_random_perturbations = "+str(num_rand_pert) +"\n" +
            "time_steps_simulation = " + str(t_steps) +"\n" + 
            "number_epochs = "+str(epochs)+"\n")
    f.write("\n")
    
    f.write("Environment parameters:"+"\n")
    
    f.write("num_agents = " + str(env_params['num_agents']) + "\n" +
            "min_allowed_distance = " + str( env_params['min_allowed_distance']) + "\n"+
            "box_multiplier = " + str( env_params['box_multiplier']) + "\n" +
            "rod_size = " + str( env_params['rod_size']) + "\n" +
            "penalty_for_being_close = " + str( env_params['penalty_for_being_close']) + "\n" +
            "alpha_angular = " + str( env_params['alpha_angular']) + "\n" +
            "beta_angular = " + str( env_params['beta_angular']) + "\n" +
            "reward_weights = " + str( env_params['reward_weights']) + "\n" +
            "delta = " + str( env_params['delta']) + "\n" +
            "std_depth = " + str( env_params['std_depth']) + "\n" +
            "min_velocity = " + str( env_params['min_velocity']) + "\n" +
            "std_straight = " + str( env_params['std_straight']) + "\n" +
            "type_behaviour = " + str( env_params['type_behaviour']) + "\n" +
            "mill_scl_r = " + str( env_params['mill_scl_r']) + "\n" +
            "mill_scl_xy = " + str( env_params['mill_scl_xy']) + "\n" +
            "mill_scl_c = " + str( env_params['mill_scl_c']) + "\n" +
            "topological_neighbors = " + str( env_params['topological_neighbors']) + "\n" +
            "num_neighbors = " + str( env_params['num_neighbors']))

    f.close()

    print('Agents behaviour: ',agents.env.type_behaviour)
    print('Type visual system: ', type_visual_system)
    print("Start training the agents with a " + policy_network_info)  
    print(stochastic_policy,stochastic_attention,symetric_policy)
    print("Sigma = "+str(sigma) + ", lrate = " + str(lrate) +", num random perturbation = "
          +str(num_rand_pert)+" timesteps of sim: " + str(t_steps))
    if type_visual_system == 'fran_model':
        print("N agents: " + str(env_params['num_agents']))
    else:
        print("N rods: " + str(agents.env.state_dim/2) + " N agents: " + str(env_params['num_agents']))
    
    start = timeit.default_timer()
    
    for epoch in range(epochs):
        
        print("Start epoch: " + str(epoch) + " with learning rate: " + str(agents.lrate) + " and sigma: " + str(agents.sigma))
        
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
            if type_visual_system == 'fran_model':
                saver(agents.policy_tensor,epoch,directory,n_layers_policy, id_tensor = 'policy_')
                saver(agents.attention_tensor,epoch,directory,n_layers_attention,id_tensor = 'attention_')
            else:
                saver(agents.tensor,epoch,directory,n_layers_policy)

            reward_file_name = os.path.join(directory,'rewards_' + str(epoch) + '.txt')
            np.savetxt(reward_file_name, np.array(reward_vector))
            std_reward_file_name = os.path.join(directory,'std_rewards_' + str(epoch) + '.txt')
            np.savetxt(std_reward_file_name, np.array(std_reward))
        
        
        
        
