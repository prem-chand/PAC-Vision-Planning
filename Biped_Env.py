#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:23:34 2020

@author: premchand
"""
import torch
import numpy as np
import matlab.engine

class Environment():
    def __init__(self, num_steps, actions, proc_num):
        # eng = matlab.engine.start_matlab()
        self.num_steps = num_steps
        # self.init_state = init_state
        self.actions = actions
        # self.p_st_foot = p_st_foot
        self.p = proc_num
        
        
        
    def compute_cost(self, policy, seed):
        eng = matlab.engine.start_matlab()
        eng.cd('/Users/premchand/Downloads/GitHub/3D biped from scratch/')
        eng.addpath(eng.genpath(eng.pwd()))
        
        params = eng.gen_params(seed, nargout = 1)
        
        eng.workspace['ws'] = eng.transpose(params['w_s'])
        eng.workspace['x0'] = params['fp']['X_star1']
        
        init_state = eng.eval('x0+ws')
        # self.init_state + params["w_s"] # add random noise to the initial state
        t = matlab.double([0])
        p_st_foot = params['p_st_foot']
        # actions = {} # define actions as an empty dictionary 
        i = 0
        is_fallen = False
        
        while(i <= self.num_steps and is_fallen == False):
            # pass the state vector to the policy 
            res = policy(torch.as_tensor(init_state).view(1,18))[0]
            # choose the largest componet of res
            res = int(res.max(0).indices)
            # pick  action according to the largest component
            turn = self.actions[str(res)] # int value of turn in degrees (hope so)            
            beta = eng.compute_beta(matlab.double([turn]), nargout = 1)
                        
            # out = eng.step2step(t, init_state, beta, p_st_foot, params, nargout = 4)
            out1 = eng.left_stance_transition(t, init_state, beta, p_st_foot, params, nargout = 4)
            if out1[3] == False:
                out2 = eng.right_stance_transition(out1[1], out1[0], beta, out1[2], params, nargout = 4)
            else:
                break
            
                
            init_state = out2[0]
            # print(out[3])
            t = out2[1]
            p_st_foot = out2[2]
            is_fallen = out2[3]
            i = i+1
        
        print('Process: {}, steps: {}'.format(self.p, i))
        cost = 0.5
        
        eng.quit()
        return cost