# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:20:13 2018

@author: Tiago Costa
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:56:31 2018

@author: Tiago Costa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:51:02 2017

@author: root
"""

import numpy as np
import timeit

#import warnings
#warnings.filterwarnings("error")

# =============================================================================
# add drag to turning - future project
# =============================================================================


class Environment:
    def __init__(self, num_agents                 = 3, 
                 min_allowed_distance             = 0.2,  
                 rod_size                         = 5,
                 penalty_for_being_close          = 20, 
                 alpha_angular                    = 1, 
                 beta_angular                     = 100,
                 reward_weights                   = np.array([2,4,1]),
                 delta                            = 1/8*np.pi,
                 std_depth                        = 0.02,    
                 min_velocity                     = 0.01,
                 std_straight                     = 0.001,
                 type_behaviour                   = 'sphere',
                 mill_scl_r                       = 2,
                 mill_scl_xy                      = 0.5,
                 mill_scl_c                       = -15,
                 ):
        
        #Type of behaviour
        if any( type_behaviour in s for s in ['sphere','tornado','schooling','milling']):
            self.type_behaviour = type_behaviour
        else:
            self.type_behaviour = 'sphere'
            
            
        #Params for computational constrain
        self.eps = np.finfo(np.float32).eps
         
        #Params of the visual system
        self.rod_size        = rod_size
        self.d_theta = delta;
        self.d_phi = delta;
        
        #Confirmation params
        self.build_rods()

        
        #Params for environment
        self.num_agents     = num_agents
        self.init_box_size  = num_agents*0.15        #box to inicialize the agents so they are close in the beggining
        
        #Params for the reset
        if self.type_behaviour == 'tornado':
            self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)*np.array([1,1,1]).astype('float32')
            self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
            self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
        elif self.type_behaviour == 'milling':
#            self.init_box_size  = num_agents*0.1 
            self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)*np.array([1,1,0.7]).astype('float32')
            self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
            self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
        else:
            self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)
            self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
            self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
            
        self.state_dim                = 2 * np.sum(self.num_rods).astype(int)
        self.state                    = np.zeros((self.num_agents,self.state_dim), dtype='float32')

        
        #Params regarding biological constrains - network outputs
        self.max_acceleration_network   = 0.15          #already multiplied by time for simplicity
        self.max_head_angle_xy          = np.pi/8
        self.max_head_angle_z           = 2*np.pi/3
        #params for physical constrains
        self.drag                       = 0.9
        self.dt                         = 0.1
        #Constants that scale actions
        self.scale_actions = np.array([self.max_acceleration_network, self.max_head_angle_xy, self.max_head_angle_z],dtype='float32')
        self.center_angle  = np.array([0, self.max_head_angle_xy/2, np.pi/2 + self.max_head_angle_z/2],dtype='float32')
      
      
        #############################- Reward params  ############################ 
        self.reward_weights = reward_weights

        # Reward params distance           
        self.min_allowed_distance        = min_allowed_distance
        self.max_closeness_allowed       = (self.rod_size-(self.min_allowed_distance))/self.rod_size
        self.penalty_for_being_close     = - penalty_for_being_close

        self.declive     = -1/3
        if self.type_behaviour == 'milling':
            self.declive     = -3
        # RP milling
        self.mill_scl_r  = mill_scl_r
        self.mill_scl_xy = mill_scl_xy
        self.mill_scl_c  = mill_scl_c

        # Reward params angular displacement      
        # Rotating            
        self.alpha_angular                    = alpha_angular
        self.beta_angular                     = beta_angular
        a_temp = np.arange(0,0.2,0.001)
        self.scalling_angular_reward          =  1/np.max(a_temp**self.alpha_angular*(1-a_temp)**(self.beta_angular-1))
        #going straight
        self.std_straight                     = std_straight

        # Reward params depth
        self.std_depth                        = std_depth**2
        
        # Reward params not moving                  
        self.min_velocity                     = min_velocity
        
    def reset(self,soft_reset):
        # Reset function to inicialize agents with random position and orientation
        # 0 - Hard reset, 1 - Soft reset
        if soft_reset == 0:
            if self.type_behaviour == 'tornado':
                self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)*np.array([1,1,1]).astype('float32')
                self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
                self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
            elif self.type_behaviour == 'milling':
                self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)*np.array([1,1,0.7]).astype('float32')
                self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
                self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
            else:
                self.reset_positions          = (np.array(np.random.rand(self.num_agents,3),dtype='float32')*self.init_box_size-self.init_box_size/2)
                self.reset_orientation_angles = np.array(np.random.rand(self.num_agents),dtype='float32')*np.pi*2
                self.reset_velocities         = np.array(np.random.rand(self.num_agents),dtype='float32')*0.2
            
        self.positions                        = np.array(self.reset_positions,dtype='float32')
        self.previous_positions               = np.array(self.positions,dtype='float32')
        
        self.orientation_angles               = np.array(self.reset_orientation_angles, dtype='float32')
        self.previous_orientation_angles      = np.array(self.orientation_angles, dtype='float32')

        #Parameters of the state
        self.velocity                         = np.array(self.reset_velocities, dtype='float32')
        self.previous_velocity                = np.array(self.reset_velocities, dtype='float32')
        
        #state reward, counter
        self.reward        = 0
        self.local_counter = 0
        self.build_state()
        return self.state


    def step(self,action):
        # Updates the positions and orientations of all agents according to an action
        #save previous head angle velocity orientation and position
        self.previous_velocity      = np.array(self.velocity,dtype='float32')        
        self.previous_positions     = np.array(self.positions,dtype='float32')
            
        #scale actions and update orientation and velocity
        self.action = self.scale_actions_to_biol_constrains(action)

        #update velocity
        self.velocity      = self.velocity + self.action[:,0]
        self.velocity      *= self.drag
        
        #update orientation
        self.orientation_angles  = (self.orientation_angles + self.action[:,1]) % (2*np.pi)
        
        
        self.positions[:,0] = self.positions[:,0] + self.velocity*self.dt*np.cos(self.orientation_angles)*(-np.sin(self.action[:,2]))
        self.positions[:,1] = self.positions[:,1] + self.velocity*self.dt*np.sin(self.orientation_angles)*(-np.sin(self.action[:,2]))
        self.positions[:,2] = self.positions[:,2] + self.velocity*self.dt*np.cos(self.action[:,2])
        self.build_state()
        self.get_reward()
        self.local_counter += 1
        
        return self.state,self.reward,self.local_counter
        
        
    def build_state(self):
        for i in range(self.num_agents):

            visual_system_state = self.get_visual_system_fish(i, memory = False)
            previous_visual_system_state = self.get_visual_system_fish(i, memory = True)
            
            self.state[i] = np.concatenate([visual_system_state, previous_visual_system_state])
      
        
    def get_visual_system_fish(self, i , memory):

        temp_state = np.zeros(np.sum(self.num_rods).astype(int), dtype='float32')
        if memory:
            focal       = self.positions[i]
            target      = np.delete(self.previous_positions,i,0)
            orientation_vector = np.array([np.cos(self.orientation_angles[i]),np.sin(self.orientation_angles[i])]).astype('float32')
        else:
            focal           = self.positions[i]
            target    = np.delete(self.positions,i,0)
            orientation_vector = np.array([np.cos(self.orientation_angles[i]),np.sin(self.orientation_angles[i])]).astype('float32')
        vector_t_f  = target - focal
        distance_f_t = np.linalg.norm(vector_t_f,axis=1);
        elevation_t = np.real(np.arccos(np.clip(vector_t_f[:,2]/(distance_f_t+self.eps),-1,1)));
        norm_xy_v = vector_t_f[:,0:2]/(np.linalg.norm(vector_t_f[:,0:2],axis = 1)+self.eps)[:,None]
        azimuth_t = np.arctan2(np.cross(orientation_vector,norm_xy_v),np.sum(norm_xy_v*orientation_vector,axis=1)  ) % (2*np.pi)      
        ind_elevation = np.round(elevation_t/self.d_theta).astype(int);
        ind_azimuth = np.remainder(np.round(azimuth_t/(2*np.pi/self.num_rods[ind_elevation])).astype(int),self.num_rods[ind_elevation]);     
        ind = np.argsort(distance_f_t)[::-1]

        temp_state[self.acc_sum[ind_elevation[ind]]+ind_azimuth[ind]] = (self.rod_size-np.minimum(distance_f_t[ind],self.rod_size))/self.rod_size
        return temp_state

              
    def get_reward(self):
        
        if self.type_behaviour == 'tornado':
            r_mean_dist_to_center, cost_distances = self.get_reward_distance_tornado()
        elif self.type_behaviour == 'milling':
            r_mean_dist_to_center, cost_distances = self.get_reward_distance_milling()
        else:
            r_mean_dist_to_center, cost_distances = self.get_reward_distance_sphere()
        if self.type_behaviour == 'schooling':
            r_angular_displacement                = self.get_reward_move_straight()
            r_movement                            = self.get_reward_velocity()
        else:
            r_angular_displacement                = self.get_reward_angular_displacement()
            r_movement                            = self.get_reward_velocity() + 0.0 * self.get_reward_turning()

        r_depth                                   = self.get_reward_depth()
        
            
#        print(r_angular_displacement,'r_angular_displacement')
#        print(r_mean_dist_to_center, 'r_mean_dist_to_center')
#        print(r_depth, 'r depth')
#        print(r_movement,'cost not moving')
#        print(cost_distances,'cost_being_close')
#        print("")
            
            
        self.reward      = np.mean(self.reward_weights[0] * r_angular_displacement +
                                   self.reward_weights[1] * r_mean_dist_to_center  +
                                   self.reward_weights[2] * r_depth +
                                   r_movement + cost_distances)/(np.sum(10 * self.reward_weights))
        
    def get_reward_velocity(self):
        return np.heaviside((np.mean(self.velocity)-self.min_velocity),0.5)-1
    
    def get_reward_turning(self):
        return -np.heaviside(-np.mean(self.action[:,1]),0.5)
        
    def get_reward_move_straight(self):
        return np.mean(np.exp(-(self.action[:,1])**2/self.std_straight))
    
    def get_reward_depth(self):
        return np.mean(np.exp(-(self.positions[:,2]-self.previous_positions[:,2])**2/self.std_depth))
    
    def get_reward_distance_sphere(self):
        #reward based on your visual system
        dist_closest_neighbour = np.max(self.state[:,:int(self.state_dim/2)],axis = 1)
        center_of_mass         = np.mean(self.positions,axis=0)
        mean_dist_to_center    = np.mean(np.linalg.norm(self.positions-center_of_mass,axis=1))
        cost_being_close       = np.mean(np.heaviside(dist_closest_neighbour-self.max_closeness_allowed,0.5)*(self.penalty_for_being_close))
        r_mean_dist_to_center  = np.maximum(self.declive*mean_dist_to_center,-5)
#        cost_being_close_center = - np.heaviside(-(np.mean(np.linalg.norm(self.positions-center_of_mass,axis=1))-0),0.5)
        return r_mean_dist_to_center, cost_being_close#, cost_being_close_center

    def get_reward_distance_tornado(self):
        #reward based on your visual system
        dist_closest_neighbour = np.max(self.state[:,:int(self.state_dim/2)],axis = 1)
        center_of_mass         = np.mean(self.positions,axis=0)
        dist_center_xy         = np.linalg.norm(self.positions[:,:2]-center_of_mass[:2],axis=1)
        dist_center_z          = np.abs(self.positions[:,2] - center_of_mass[2])
        
        cost_being_close       = np.mean(np.heaviside(dist_closest_neighbour-self.max_closeness_allowed,0.5)*(self.penalty_for_being_close))
        r_dist_center_xy       = np.maximum(self.declive*np.mean(dist_center_xy),-5)
        r_dist_center_z        = np.maximum(self.declive*np.mean(dist_center_z),-5)
        
        return 0.7*r_dist_center_xy + 0.3*r_dist_center_z, cost_being_close


    def get_reward_distance_milling(self):
        dist_closest_neighbour = np.max(self.state[:,:int(self.state_dim/2)],axis = 1)
        center_of_mass         = np.mean(self.positions,axis=0)
        dist_center_xy         = np.linalg.norm(self.positions[:,:2]-center_of_mass[:2],axis=1)
        dist_center_z          = np.abs(self.positions[:,2] - center_of_mass[2])
        cost_being_close       = np.mean(np.heaviside(dist_closest_neighbour-self.max_closeness_allowed,0.5)*(self.penalty_for_being_close))
        
        r_dist_center_xy  = self.mill_scl_r+\
                            self.mill_scl_c*np.exp(-1/(1-np.clip(((self.mill_scl_xy+0.25)*dist_center_xy)**2,0,0.999)))+\
                            np.clip(-1/5*np.maximum(self.mill_scl_xy*dist_center_xy-1,0),-10,0)
        r_dist_center_z   = np.maximum(self.declive*np.mean(dist_center_z),-5)
        return 0.5*np.mean(r_dist_center_xy) + 0.5*r_dist_center_z, cost_being_close 
#        return r_dist_center_xy

    def get_reward_angular_displacement(self):
        angular_displacement = self.get_angular_displacement()
        angular_d_cliped = np.clip(angular_displacement,0,1)
#        print(angular_displacement)
        return np.mean(self.scalling_angular_reward*(angular_d_cliped**self.alpha_angular*(1-angular_d_cliped)**(self.beta_angular-1)))
    
    
    def get_angular_displacement(self):
        #Angular displacement in the XY plane
        previous_center_of_mass = np.mean(self.previous_positions,axis=0)[:2]
        previous_position_vector_center_of_mass = np.array(self.previous_positions[:,:2]-previous_center_of_mass) + self.eps
        position_vector_center_of_mass          = np.array(self.positions[:,:2]-previous_center_of_mass) + self.eps
        y = np.cross(previous_position_vector_center_of_mass,position_vector_center_of_mass)
        x = np.sum(previous_position_vector_center_of_mass*position_vector_center_of_mass,axis=1)        
        angular_displacement = np.arctan2(y,x)
        return angular_displacement

    def scale_actions_to_biol_constrains(self,a):
        action = a*self.scale_actions-self.center_angle
        return action


    def build_rods(self):
        self.elevation = np.arange(0,np.pi + self.eps,self.d_theta).astype('float32')
        self.num_elevation = len(self.elevation)

        cos_theta = np.cos(self.elevation - np.pi/2)
 
        perimeter = 2*np.pi*cos_theta
        
        self.num_rods     = np.floor((perimeter/self.d_phi))
        self.num_rods[0]  = 1
        self.num_rods[-1] = 1
        self.num_rods     = self.num_rods.astype(int)
        self.azimuth      = [[] for i in range(self.num_elevation)]
        self.angles       = np.zeros((2,np.sum(self.num_rods).astype(int)))
        count = 0
        for i in range(self.num_elevation):
            if self.num_rods[i] > 0:
                self.azimuth[i] = np.arange(0,self.num_rods[i]).astype('float32')*2*np.pi/self.num_rods[i]
            else:
                self.azimuth[i] = np.array([0]).astype('float32')
            for j in range(len(self.azimuth[i])):
                self.angles[:,count] = np.array([self.elevation[i],self.azimuth[i][j]])
                count += 1
        self.rods = np.array([np.sin(self.angles[0])*np.cos(self.angles[1]),np.sin(self.angles[0])*np.sin(self.angles[1]),np.cos(self.angles[0])])   
        self.acc_sum = np.concatenate((np.array([0]),np.cumsum(self.num_rods).astype(int)))

        l = int(len(self.elevation)-1)
        a = np.linalg.norm(self.rods.T-self.rods[:,self.acc_sum[l//2]+0],axis=1)
        b = np.linalg.norm(self.rods.T-self.rods[:,self.acc_sum[l//2]+l],axis=1)
        
        
        self.inds_milling   = np.concatenate((np.where(a< 0.8)[0],np.where(b < 0.8)[0]))
        self.n_inds_milling =len(self.inds_milling)
        
    def force_state(self,position_vector,velocity_vector,orientation_angles):
        self.positions           = np.array(position_vector)
        self.previous_positions  = np.array(position_vector)
        
        self.velocity            = np.array(velocity_vector)
        self.previous_velocity   = np.array(velocity_vector)
        
        self.orientation_angles          = np.array(orientation_angles)
        self.previous_orientation_angles = np.array(orientation_angles)

        self.build_state()
       

    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(1)  
    start = timeit.default_timer()
    num_agents = 3
    num_rods = 500
    env_params = {'num_agents'                      : 3, 
                 'min_allowed_distance'             : 0.3,  
                 'rod_size'                         : 5,
                 'penalty_for_being_close'          : 20, 
                 'alpha_angular'                    : 1, 
                 'beta_angular'                     : 30,
                 'reward_weights'                   : np.array([2,1,0.5]),
                 'delta'                            : 1/12*np.pi,
                 'std_depth'                        : 0.025,    
                 'min_velocity'                     : 0.05,
                 'std_straight'                     : 0.001,
                 'type_behaviour'                   : 'milling',
                 'mill_scl_r'                       : 2,
                 'mill_scl_xy'                      : 0.6,
                 'mill_scl_c'                       : -35,
                 }

    env= Environment(**env_params)
#    start = timeit.default_timer()
#
    state = env.reset(0)
# =============================================================================
    
    
    
    
#    print(np.max(env.positions,0)-np.min(env.positions,0))
#    print("")
#    state = env.reset(1)
#    print(np.max(env.positions,0)-np.min(env.positions,0))
#    print("")
#    state = env.reset(0)
#    print(np.max(env.positions,0)-np.min(env.positions,0))
#    print("")
#    stop = timeit.default_timer()
#
#    print(stop - start)
#    acc = 0.3
#    v_eq = acc*env.drag*env.max_acceleration_network/(1-env.drag)
#    position_vector= np.array([[1,0,0],[0,0,0],[-1,0,0]],dtype = 'float32')
#    velocity_vector = np.array([v_eq,0,v_eq])
#    orientation_angles = np.zeros((num_agents),dtype = 'float32')
#    orientation_angles = np.array([np.pi/2,0,3/2*np.pi])
#    env.force_state(position_vector,velocity_vector,orientation_angles)
#    print(state[:,:320])
# =============================================================================
#
#    dt  = 0.2
#    lim = 7
#    size = int(2*lim/dt)+1
#    x = np.arange(-lim,lim + 10E-6, dt).reshape(1,size)
#    #y = np.arange(-2,2,0.05).reshape([400,400])
#    y = np.arange(-lim,lim + 10E-6, dt).reshape(size,1)
#    
#    xM = np.matlib.repmat(x, size, 1).reshape(-1).reshape(size*size,1)
#    yM =  np.matlib.repmat(y, 1, size).reshape(-1).reshape(size*size,1)
#    zM = np.zeros((size*size,1))
#    pos = np.concatenate((xM,yM,zM),axis = 1)
#    env.force_state(pos,velocity_vector,orientation_angles)
#    r = env.get_reward_distance_milling()
#    fig = plt.figure(figsize =[15,10])
#    ax = fig.add_subplot(111, projection = '3d')
#    size_dx = 7
#    size_dy = 7
#    size_dz = 3
#    ax.set_xlim([-size_dx,size_dx])
#    ax.set_ylim([-size_dy,size_dy])
#    ax.set_zlim([-size_dz,size_dz])
#    
#    ax.scatter(xM, yM, r, c = 'r',alpha = 0.05)
#    
#    d = ((xM**2+yM**2)**0.5).reshape(-1)
#    fig = plt.figure()
#    ax = fig.add_subplot(121)
#    plt.scatter(range(len(d[np.abs(r-2)<0.001])),d[np.abs(r-2)<0.001])
#    ax = fig.add_subplot(122)
#    plt.scatter(range(len(d[(np.abs(r-0)<0.1) & (d<1)])),d[(np.abs(r-0)<0.1) & (d<1)])
#    plt.show()
#    
#    fig = plt.figure(figsize =[15,10])
#    ax = fig.add_subplot(111, projection = '3d')
#    size_dx = 7
#    size_dy = 7
#    size_dz = 3
#    ax.set_xlim([-size_dx,size_dx])
#    ax.set_ylim([-size_dy,size_dy])
#    ax.set_zlim([-size_dz,size_dz])
#    ax.scatter(xM[np.abs(r-2)<0.001], yM[np.abs(r-2)<0.001], r[np.abs(r-2)<0.001], c = 'r',alpha = 0.05)
#    ax.scatter(xM[r<-1], yM[r<-1], r[r<-1], c = 'r',alpha = 0.05)
#    
#    plt.show()
#
#    plt.scatter(xM[r<-4], yM[r<-4])

#d = np.array([0.1,0,0.1])
# =============================================================================
    # check rods
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection = '3d')
#    size_dx = 3
#    size_dy = 3
#    size_dz = 3
#    ax.set_xlim([-size_dx,size_dx])
#    ax.set_ylim([-size_dy,size_dy])
#    ax.set_zlim([-size_dz,size_dz])
##    rods = np.array([np.sin(env.angles[0])*np.cos(env.angles[1]),np.sin(env.angles[0])*np.sin(env.angles[1]),np.cos(env.angles[0])])  
#    ax.scatter(env.rods[0], env.rods[1], env.rods[2], c = 'r',alpha = 0.2)
#
#
##    ax.scatter(env.rods[0,env.acc_sum[4]+0], env.rods[1,env.acc_sum[4]+0], env.rods[2,env.acc_sum[4]+0], c = 'b')
##    ax.scatter(env.rods[0,env.acc_sum[4]+8], env.rods[1,env.acc_sum[4]+8], env.rods[2,env.acc_sum[4]+8], c = 'b')
#    ax.scatter(env.rods[0,env.inds_milling ], env.rods[1,env.inds_milling ], env.rods[2,env.inds_milling ], c = 'b')

# =============================================================================

#    Check ray casting!
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    fishid = 2
#    fish_pos     = env.positions[fishid]
#    fish_ori     = env.orientation_angles[fishid]
#    state_teste = env.get_visual_system_fish(fishid, memory = False)
#    inds = np.argwhere(state_teste>0)
#    plot_angles = env.angles[:,inds.reshape(-1)]
#    
#    points_plot = np.zeros((3,np.sum((state_teste>0)[fishid])))
#    
    
    
    
    
#    for i in range(1):
#        count = 0
#        for j in range(num_rods):
#            if (state_teste>0)[fishid,j]==1:
#                points_plot[:,count] = (5-5*state_teste[fishid,j])*env.visual_system[fishid,:,j]+fish
##                count +=1
#    ax.scatter(points_plot[0,:],points_plot[1,:],points_plot[2,:],alpha=0.4)
#    for i in range(num_agents):
#        otherfish = env.positions[i]
#        ax.scatter(otherfish[0],otherfish[1],otherfish[2],s=200)
# =============================================================================

    
    #############################teste movement##################################
##    env.reset(1)
##    print(env.reset_positions)
##    print(env.reset_orientation_angles)
##    print("")
#    print("positions after force state")
##    print(env.positions)
##    print(env.orientation_angles)
#    print("")
#
#    print("Actions")
#    tr = 0
#    start = timeit.default_timer()
#    a=np.array([[acc,0.6,1],[0,0.5,0.5],[acc,0.6,0]])
#
#    for i in range(1):
##        a=np.array([[1,0.7,0.5],[0,0,0],[1,0.7,0.5]])
#        state,r,_ = env.step(a)
##        print(env.velocity)
#        print(-np.sin(env.action[:,2]))
#        print(np.cos(env.action[:,2]))
#
##        a= np.random.rand(num_agents,3)
#        tr += r
###        print(r)
#        f1, ax1 = plt.subplots()
#        plt.xlim([-5,5])
#        plt.ylim([-5,5])
#            #f1.pause(0.5)
#        ax1.scatter(env.previous_positions[:,0],env.previous_positions[:,1])
#        ax1.scatter(env.positions[:,0],env.positions[:,1])
#        previous_center_of_mass = np.mean(env.previous_positions,axis=0)
##        previous_center_of_mass = np.array([0,0])
#        ax1.scatter(previous_center_of_mass[0],previous_center_of_mass[1])
#        plt.show()
#        print("")
##        print(env.positions)
#        print("")
#        plt.pause(0.05)
##    print("")
#    print(tr)
#    stop = timeit.default_timer()
#    print(stop-start)
#    print(env.velocity)
#        print(env.previous_positions)
#        print("")
#        print(env.positions)
#    
    ##########################################################################
#    a=np.array([[1,1,0],[1,0,1],[1,0.5,0.5]])
#    state,r,_,d = env.step(a)
#    print("original positions")
#    print(env.reset_positions)
#    print(env.previous_orientation_angles)
#    print("")
#    print("positions after moving")
#    print(env.positions)
#    print(env.orientation_angles)
#    state = env.reset(40)
#    print("")
#
#    print("soft reset")
#    print(env.reset_positions)
#    print(env.reset_orientation_angles)
#    print("")
#
#    print("positions")
#    print(env.positions)
#    print(env.orientation_angles)
#    state = env.reset(0)
#    print("")
#
#    print("hard reset")
#    print(env.reset_positions)
#    print(env.reset_orientation_angles)
#    print("")
#
#    print("positions")
#    print(env.positions)
#    print(env.orientation_angles)
    ##########################################################################

#    a=np.array([[0.2,0.5],[1,0.3],[0,0.7],[0,0]],dtype='float32')
##    print(env.state[2])
##
#    for i in range(1):   
#        state,r,_,d = env.step(a)
#
##        print("")
#        print("")
    
    
# =============================================================================
