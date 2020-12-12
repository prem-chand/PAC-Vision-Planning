#!/usr/bin/env python3

import torch.multiprocessing as mp
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
# import matlab.engine


class Compute_Loss:

    def __init__(self, num_trials, num_cpu=1, num_gpu=1, start_seed=0):
        # self.cores = mp.cpu_count()
        self.cores = num_cpu
        self.batch = [0 for _ in range(self.cores)]  # give each core the correct number of processes + data
        for i in range(num_trials):
            self.batch[i % self.cores] += 1
        self.num_trials = num_trials
        self.num_gpu = num_gpu
        self.start_seed = start_seed

    def compute(self, itr_num, params, mu, std):

        example = params['example']
        
        # import policy based on the example
        if example == 'quadrotor':
            from policy.quad_policy import Policy, Filter
            policy = Policy()
            DepthFilter = Filter()
            policy.share_memory()
            DepthFilter.share_memory()
            nets = [policy, DepthFilter]
        elif example == 'minitaur':
            from policy.minitaur_policy import Policy
            policy = Policy()
            policy.share_memory()
            nets = [policy]            
        elif example == 'biped':
            from policy.biped_policy import Policy
            policy = Policy()
            policy.share_memory()
            nets = [policy]
            
        # Need this start_method for parallelizing Pytorch models
        mp.set_start_method('forkserver', force=True)
        process = []
        batch = self.batch
        manager = mp.Manager()
        rd = manager.dict()
        pos = self.start_seed
        torch_pos = 0
        device = [torch.device('cuda:'+str(i)) for i in range(self.num_gpu)]

        # Assumption: num_cpu_cores >= num_gpu
        if self.num_gpu>0:
            device = [torch.device('cuda:'+str(i)) for i in range(self.num_gpu)]
            # Assumption: num_cpu_cores >= num_gpu
            device_list = [0] * self.cores
            for i in range(self.cores):
               # device_counter[i % self.num_gpu] += 1
               device_list[i] = i % self.num_gpu
        else:
            device = [torch.device('cpu')]
            device_list = [0] * self.cores

        for j in range(self.cores):
            
            # Generate seeds at random with no regard to repeatability
            # torch_seed = [self.new_seed() for i in range(batch[j])]
            
            # Generate the same seeds for every instance of training for comparison
            # of hyperparameters; the seeds differ with the iteration number.
            torch_seed = list(range(itr_num*self.num_trials + torch_pos, 
                                    itr_num*self.num_trials + torch_pos + batch[j]))

            # Fixing the np_seed fixes the enviroment
            np_seed = list(range(pos,pos+batch[j]))

            if example == 'quadrotor':
                process.append(mp.Process(target=self.quadrotor_thread, args=(params, nets, device[device_list[j]],
                                                                              mu, std, batch[j], np_seed, torch_seed, 
                                                                              rd, j)))
            elif example == 'minitaur':
                process.append(mp.Process(target=self.minitaur_thread, args=(params, nets, device[device_list[j]],
                                                                             mu, std, batch[j], np_seed, torch_seed, 
                                                                             rd, j)))
            
            elif example == 'biped':
                process.append(mp.Process(target=self.biped_thread, args=(params, nets, device[device_list[j]],
                                                                             mu, std, batch[j], np_seed, torch_seed, 
                                                                             rd, j)))

            pos += batch[j]
            torch_pos += batch[j]
            process[j].start()

        for j in range(self.cores):
            process[j].join()

        # Collect the epsilons along with cost (how to keep them separate from other environments?)
        grad_mu = torch.zeros(mu.numel())
        grad_logvar = torch.zeros(std.numel())
        emp_cost = []
        for i in range(self.cores):
            grad_mu += rd[i][0]
            grad_logvar += rd[i][1]
            
            # torch.cat misbehaves when there is a 0-dim tensor, hence view
            emp_cost.extend(rd['costs'+str(i)].view(1,rd['costs'+str(i)].numel()))

        emp_cost = torch.cat(emp_cost)

        grad_mu /= self.num_trials
        grad_logvar /= self.num_trials
        emp_cost = emp_cost.sum()/self.num_trials

        return emp_cost, grad_mu, grad_logvar

    @staticmethod
    def new_seed():
        return int(2 ** 32 * np.random.random_sample())
    
    @staticmethod
    def biped_thread(params, nets, device, mu, std, batch_size, np_seed, 
                         torch_seed, rd, proc_num):
        grad_method = params['grad_method']
        num_policy_eval = params['num_policy_eval']
        num_steps = params['num_steps']
        # init_state = params['init_state']
        actions = params['actions']
        # p_st_foot = params['p_st_foot']
        
        policy = nets[0]
        
        from ES_grad import compute_grad_ES
        from envs.Biped_Env import Environment
        
        env = Environment(num_steps, actions, proc_num)
        
        # creating objects
        policy_eval_costs = torch.zeros(num_policy_eval*2)
        grad_mu = torch.zeros(mu.numel())
        grad_logvar = torch.zeros(std.numel())
        batch_costs = torch.zeros(batch_size)
        
        # print(torch_seed)
        # Generate epsilons in here and compute multiple runs for the same environment
        for i in range(batch_size):
            torch.manual_seed(torch_seed[i])
            epsilon = torch.randn((num_policy_eval, mu.numel()))
            epsilon = torch.cat([epsilon, -epsilon], dim=0)
            # if i>0:
                
            np.random.seed(np_seed[i])
            # env.generate_trajectory()
            
            for j in range(num_policy_eval*2):
                if j == num_policy_eval*2:
                    policy_params = mu
                else:
                    policy_params = mu + std*epsilon[j,:]

                policy_params = policy_params.to(device)
                
                # LOAD POLICY_PARAMS
                count = 0
                for p in policy.parameters():
                    num_params_p = p.data.numel()
                    p.data = policy_params[count:count+num_params_p].view(p.data.shape)
                    count+=num_params_p
                    
                cost = env.compute_cost(policy, torch_seed[i])
                
                policy_eval_costs[j] = torch.Tensor([cost])

            batch_costs[i] = policy_eval_costs.mean()

            grad_mu_temp, grad_logvar_temp = compute_grad_ES(policy_eval_costs-policy_eval_costs.mean(), 
                                                             epsilon, std, grad_method)
                
            grad_mu += grad_mu_temp
            grad_logvar += grad_logvar_temp

        # Gradient is computed for 1-loss, so return its negation as the true gradient
        rd[proc_num] = [grad_mu, grad_logvar]

        # Return the sum of all costs in the batch
        rd['costs'+str(proc_num)] = batch_costs
        # env.quit_matlab_engine()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        