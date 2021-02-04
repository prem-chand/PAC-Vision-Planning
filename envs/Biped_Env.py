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
    def __init__(self, num_steps, actions, proc_num, device = "cuda"):
        self.num_steps = num_steps
        # self.init_state = init_state
        self.actions = actions # dictionary of indexed primitives (in degs)
        # self.p_st_foot = p_st_foot
        self.p = proc_num
        self.device = device
        
        
    def compute_cost(self, policy, seed, torch_seed):
        try:
            eng = matlab.engine.start_matlab()
        except matlab.engine.EngineError: 
            eng = matlab.engine.start_matlab()
        eng.cd('/home/rhaegar/Github/3D_biped_updated/')
        eng.addpath(eng.genpath(eng.pwd()))
        
        params = eng.gen_params(seed, nargout = 1)
        
        x2m = params['x0']
        init_state = params['init_state']
        # self.init_state + params["w_s"] # add random noise to the initial state
        t = matlab.double([0])
        p_st_foot = params['p_st_foot']
        # actions = {} # define actions as an empty dictionary 
        k = 1 # step counter
        # d = 0
        cost1, cost2, cost = 0, 0, 0
        max_dev = 0
        is_fallen = False
        prim_seq = []
        
        while(k <= self.num_steps and is_fallen == False):
            # pass the state vector to the policy 
            res = policy(torch.tensor(init_state).view(1,20).to(self.device))[0]
            # choose the largest componet of res
            res = int(res.max(0).indices)
            # pick  action according to the largest component
            turn = self.actions[str(res)] # int value of turn in degrees (hope so) 
            prim_seq.append(turn)
            beta = eng.compute_beta(matlab.double([turn]), nargout = 1)
            
            out1 = eng.left_stance_transition(t, x2m, beta, p_st_foot, params, nargout = 5)
            x1m = out1[0]
            te1 = out1[1]
            p_stance_footR = out1[2]
            is_fallen = out1[3]
            simL = out1[4]
            t_float = np.float32(t)
            if is_fallen == False and (te1-t_float) > 0.1:
                Fxl = simL['Fx']
                Fyl = simL['Fy']
                cost1 += simL['cost']
                max_dev = max(max_dev,simL['max_dev'])
                out2 = eng.right_stance_transition(te1, x1m, beta, p_stance_footR, params, nargout = 5)
                x2m = out2[0]
                te2 = out2[1]
                p_stance_footL = out2[2]
                dist = np.linalg.norm(p_stance_footR)
                is_fallen = out2[3]
                simR = out2[4]
                if is_fallen == False and (te2 - te1) > 0.1:
                    Fxr = simR['Fx']
                    Fyr = simR['Fy']
                    cost2 += simR['cost']
                    max_dev = max(max_dev,simR['max_dev'])
                    dist = np.linalg.norm(p_stance_footL)
                    
                    Fx = Fxl + Fxr
                    Fy = Fyl + Fyr
                else:
                    cost2 += 0
                    is_fallen = True
                    break
            else:
                is_fallen = True
                cost1 += 0
                break
            init_state = np.concatenate((x2m, Fx, Fy) ,axis = None)
            init_state = np.float32(init_state)
            t = te2
            p_st_foot = p_stance_footL
            cost = (cost1 + cost2)/k**2
            k = k+1
        
        
        # if k<=25:
        #     max_dev = 1
        print('Policy seed: {}, Env: {}, steps: {}, dist: {:.3f}, max_dev: {:.3f}, cost: {:.3f}'.format(seed, 
                                                        torch_seed, k, dist, max_dev,
              # d, 
               # cost))
               max_dev/dist))
        
        
        eng.quit()
        # return cost
        return max_dev/dist