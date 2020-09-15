#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:47:47 2020

@author: premchand
"""

import torch
# import torch.nn as nn
# import math
# from os.path import join
# import scipy.io as io
import numpy as np
# from policy.biped_policy import Policy
import matlab.engine
# eng = matlab.engine.start_matlab()

# eng.cd('/Users/premchand/Downloads/GitHub/PAC_biped_matlab')   
# eng.addpath(eng.genpath(eng.pwd()))

class Environment():
    def __init__(self, t0, num_steps, init_state, Fx0, Fy0, p_stance_foot0, proc_num):
        self.num_steps = num_steps
        self.init_state = init_state
        self.Fx0 = Fx0
        self.Fy0 = Fy0
        self.t0 = t0
        self.p_stance_foot0 = p_stance_foot0
        self.s = np.random.randint(1000,9999)
        self.p = proc_num
        # eng = matlab.engine.start_matlab()
        
    def compute_cost(self, policy, seed):
        # proc_id = 'p'+ self.p
        eng = matlab.engine.start_matlab()
        # eng = matlab.engine.start_matlab(background=True)
        # print("Engne ID: ", eng)
        
        
        eng.cd('/Users/premchand/Downloads/GitHub/PAC_biped_matlab')   
        eng.addpath(eng.genpath(eng.pwd()))
        
        eng.workspace['s'] = seed
        eng.sim_noise0(nargout = 0)
        
        actions = {}
        actions[self.p] = []
        
        while (eng.workspace['i'] <= self.num_steps):
            x0 = eng.workspace['ye2_u']
            speed_vec = eng.workspace['speed_vector']
            int_F_x = eng.workspace['int_F_x']
            int_F_y = eng.workspace['int_F_y']
            x1 = torch.as_tensor(x0)
            speed_v = torch.as_tensor(speed_vec)
            F_x = torch.as_tensor([[int_F_x]])
            F_y = torch.as_tensor([[int_F_y]])
            inputs = torch.cat((x1, speed_v, F_x, F_y),dim = 0)
            inputs = torch.transpose(inputs,0,1)
            res = policy(inputs)[0]
            res1 = int(res.max(0).indices + 1)              
            # print("Process num:",self.p,"action: ",res1)
            actions[self.p].append(res1)
            # print("Process num:", self.p, "Engine ID: ", eng, "policy: ", actions[self.p])

            switch_count = int(eng.workspace['switch_count'])
            if eng.workspace['i'] == eng.workspace['vel_switch'][0][switch_count-1]:
                # print("te2: ",eng.workspace['te2'])
                # print("te2: ", eng.workspace['te2'], "engine: ", eng)
                # print("t_switch_vel: ",eng.workspace['t_switch_vel'])
                eng.workspace['t_switch_vel'] = eng.workspace['t_switch_vel'] + eng.workspace['t_start']
                eng.workspace['t_start'] = matlab.double([0])
                eng.workspace['pL_start'] = eng.workspace['simout']['lead'][-1]        
        
                switch_count = int(eng.workspace['switch_count'])
                eng.workspace['switch_count'] = switch_count + 1
                vL = (np.array(eng.workspace['vL_desired'])[:,eng.workspace['switch_count'] - 1]).tolist()
                eng.workspace['vL'] = matlab.double(vL)
        
            if eng.workspace['i'] > 1:
                # print(eng.workspace['actions'][0][res1-1])
                eng.workspace['turn'] = eng.workspace['actions'][0][res1-1]
        
            eng.sim_noise1(nargout = 0)
            if eng.size(eng.workspace['te1'],1) == 1 and eng.workspace['delta_t'] >= 0.3:            
                eng.sim_noise2(nargout = 0)
                if eng.size(eng.workspace['te2'],1) == 1 and eng.workspace['delta_t'] >= 0.3:
                    eng.sim_noise3(nargout = 0)
                else:
                    # eng.workspace['cost'] = 5000/eng.workspace['i']
                    # eng.workspace['cost'] = 50
                    break
            else:
                # eng.workspace['cost'] = 5000/eng.workspace['i']
                # eng.workspace['cost'] = 50
                break
                    
            # eng.simulation1(nargout = 0)
            # if eng.size(eng.workspace['te1'],1) == 0:
            #     eng.workspace['cost'] = 1000
            #     break
                
            # eng.simulation2(nargout = 0)
            # if eng.size(eng.workspace['te2'],1) == 0:
            #     eng.workspace['cost'] = 1000
            #     break
            
            
            # eng.simulation3(nargout = 0)
            
            # data.append(eng.workspace)
            eng.workspace['i'] = eng.workspace['i'] + 1
            # "cost: ", round(eng.workspace['cost'],4),
        # print("Process: ", self.p, "steps: ", int(eng.workspace['i']-1), "tracking error: ", round(eng.workspace['costT'],3))
        print('Process: {}, steps: {}, pos tracking: {:.3f}, vel tracking: {:.3f}, max force: {:.3f}'.format(self.p, 
                        int(eng.workspace['i']-1), eng.workspace['costT'], eng.workspace['costV'], eng.workspace['costF']))
        # print('Process: {}, steps: {}, tracking error: {:.3f}, max force: {:.3f}'.format(self.p, int(eng.workspace['i']-1), eng.workspace['costT'],eng.workspace['costF']))
        cost = eng.workspace['cost']
        # eng.quit()
        return cost
        
        
    def generate_trajectory(self):
        
        eng.cd('/Users/premchand/Downloads/GitHub/PAC_biped_matlab/trajectory')        
        eng.gen_vL(self.s,nargout = 0) 
        
    # def quit_matlab_engine(self):
    #     eng.quit()