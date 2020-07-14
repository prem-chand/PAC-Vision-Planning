#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:25:16 2020

@author: premchand
"""

import torch
import torch.nn as nn
from os.path import join
import numpy as np
# from policy.biped_policy import BipedPolicy
import matlab.engine
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

# eng = matlab.engine.start_matlab()
# eng.cd('/Users/premchand/Downloads/GitHub/3D-biped/old-code/executable/To python')
# data = eng.get_init_data(nargout=11)

class Environment():
    def __init__(self,t0,num_steps,init_state,Fx0,Fy0,p_stance_foot0):
        self.num_steps = num_steps
        self.init_state = init_state
        self.Fx0 = Fx0
        self.Fy0 = Fy0
        self.t0 = t0
        self.p_stance_foot0 = p_stance_foot0
        self.eng = matlab.engine.start_matlab()
        self.s = np.random.randint(1000,5000)
        # self.eng = p
        
    @profile
    def compute_cost(self, policy):
        
        cost0 = 0
        x0 = self.init_state
        int_F_x = self.Fx0
        int_F_y = self.Fy0
        t0 = self.t0
        p_stance_foot = self.p_stance_foot0
        s = self.s
        
        
        self.eng.cd('/Users/premchand/Downloads/GitHub/3D-biped/old-code/executable/To python')
        for i in range(self.num_steps):
            x1 = torch.as_tensor(x0)
            inputs = torch.cat((x1,int_F_x,int_F_y),dim = 0)
            inputs = torch.transpose(inputs,0,1)
            res = policy(inputs)[0]
            prim_idx = int(res.max(0).indices + 1)
            # print(t0)
            step_data = self.eng.simulate_step(t0,x0,prim_idx,p_stance_foot,s,nargout = 7)
            next_state = step_data[0]
            F_x = step_data[1]
            F_y = step_data[2]
            p_stance_foot = step_data[3]
            te = step_data[4]
            cost1 = step_data[5]
            is_fallen = step_data[6]
            if is_fallen:
                break
            cost0 = max(torch.as_tensor(cost0),torch.as_tensor(cost1))
            
            
            t0 = te
            x0 = next_state
            int_F_x = torch.as_tensor([[F_x]])
            int_F_y = torch.as_tensor([[F_y]])

        return cost0
        
    def generate_trajectory(self):
        self.eng.cd('/Users/premchand/Downloads/GitHub/3D-biped/old-code/executable/To python/trajectory_gen')
        
        traj = self.eng.gen_traj(self.s)
        
        return traj
        
        
        
        
        
        
        