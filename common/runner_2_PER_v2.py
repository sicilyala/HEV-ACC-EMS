from tqdm import tqdm  # 进度条
import os
import torch
import numpy as np
import scipy.io as scio
from numpy.random import normal  # normal distribution
from common.brain_PER import Brain
from common.Priority_Replay import Memory_PER


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.Memory = Memory_PER(args)
        self.agents = self._init_agents()
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.save_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
    
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Brain(i, self.args, load_or_not=self.args.load_or_not)
            agents.append(agent)
        return agents
    
    def _update_mean_std(self, agent_id, value_dict, s_dict):
        for value in value_dict:
            mean_tmp = self.env.agents[agent_id].mean_dict[value]*(1-self.args.miu)+np.mean(s_dict[value])*self.args.miu
            self.env.agents[agent_id].mean_dict[value] = mean_tmp
            std_tmp = self.env.agents[agent_id].std_dict[value]*(1-self.args.miu)+np.std(s_dict[value])*self.args.miu
            self.env.agents[agent_id].std_dict[value] = std_tmp
    
    def seperate_transitions(self, transitions):
        # handle transitions        time cost
        a0 = self.args.obs_shape[0]
        b0 = self.args.action_shape[0]
        a1 = self.args.obs_shape[1]
        b1 = self.args.action_shape[1]
        transitions = torch.as_tensor(transitions, dtype=torch.float32)
        transitions_0 = transitions[:, :a0+b0+1+a0]
        transitions_1 = transitions[:, a0+b0+1+a0:]
        s0 = transitions_0[:, :a0]       # tensor list
        act0 = transitions_0[:, a0:a0+b0]
        r0 = transitions_0[:, a0+b0:a0+b0+1]
        s_0 = transitions_0[:, a0+b0+1: a0+b0+1+a0]
        s1 = transitions_1[:, :a1]
        act1 = transitions_1[:, a1:a1+b1]
        r1 = transitions_1[:, a1+b1:a1+b1+1]
        s_1 = transitions_1[:, a1+b1+1: a1+b1+1+a1]
        return s0, act0, r0, s_0, s1, act1, r1, s_1
        
    def run(self):
        """DDPG style"""
        average_reward = []  # average_reward of each episode
        driver_average_reward = []
        energy_average_reward = []
        noise_decrease = False
        change_lr_flag = False
        # init_noise = [self.args.init_noise_0, self.args.init_noise_1]
        init_noise = self.args.init_noise
        DONE = {}
        c_loss_average = {'c_loss_0': [], 'c_loss_1': []}
        a_loss_average = {'a_loss_0': [], 'a_loss_1': []}
        lra = {'lra_0': [], 'lra_1': []}
        lrc = {'lrc_0': [], 'lrc_1': []}
        fuel = []  # fuel cost of 100 km of each episode
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            if noise_decrease:
                init_noise *= self.args.noise_discount_rate
                # init_noise[1] *= self.args.noise_discount_rate
            
            episode_reward = []  # sum of reward of a single episode
            driver_reward = []
            energy_reward = []
            episode_step = 0
            c_loss_one_episode = {0: [], 1: []}
            a_loss_one_episode = {0: [], 1: []}
            info = []
            colli_times = -1
            f_travel = 0
            feul_cost = 0
            episode_info0 = {'spd': [], 'acc': [], 'distance': [], 'leading_x': [], 'follow_x': [],
                             'collision': [], 'TTC': [], 'jerk_value': [],
                             'r_jerk': [], 'r_speed': [], 'r_safe': [], 'driver_reward': []}
            episode_info1 = {'W_eng': [], 'T_eng': [], 'P_eng': [], 'fuel_eff': [],
                             'W_ISG': [], 'T_ISG': [], 'P_ISG': [], 'Gen_eff': [],
                             'W_mot': [], 'T_mot': [], 'P_mot': [], 'Mot_eff': [],
                             'fuel_consumption': [], 'T_axle': [], 'W_axle': [],
                             'SOC': [], 'SOH': [], 'cell_power_out': [], 'P_batt': [],
                             'cell_OCV': [], 'cell_Vt': [], 'cell_V_3': [],
                             'I': [], 'I_c': [], 'tep_a': [], 'dsoh': [],
                             'fuel_cost': [], 'battery_reward': [], 'energy_reward': [],
                             'delta_SOC': [], 'cost_SOC': [], 'cost_SOH': [], 'cost_power': []}
            while episode_step < self.episode_step:
                with torch.no_grad():
                    all_actions = []
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action2(state[agent_id])
                        action = np.clip(normal(action, init_noise), -1, 1)
                        all_actions.append(action)
                    # action0 = self.agents[0].select_action2(state[0])
                    # # action0 = np.clip(normal(action0, init_noise[0]), -1, 1)  # acc
                    # action1 = self.agents[1].select_action2(state[1])
                    # action1 = np.clip(normal(action1, init_noise), -1, 1)  # engine power
                    # # action1 = np.clip(normal(action1, init_noise[1]), -1, 1)  # engine power
                    # all_actions = [action0, action1]       # attention: store action1
                state_next, reward, done, info = self.env.step(all_actions)
                tt = []
                for i in range(self.args.n_agents):
                    tt.append(np.concatenate((state[i], all_actions[i], reward[i], state_next[i]), axis=0))
                transition = np.concatenate((tt[0], tt[1]), axis=0)
                self.Memory.store(transition)
                if any(done):
                    # print('SOC failure in step %d of episode %d'%(episode_step, episode))
                    if episode not in DONE.keys():
                        DONE.update({episode: episode_step})
                    # break
                state = state_next
                # data save
                episode_reward.append(reward)  # reward: [r1, r2]
                driver_reward.append(reward[0])
                energy_reward.append(reward[1])
                for key in episode_info0.keys():
                    episode_info0[key].append(info[0][key])
                for key in episode_info1.keys():
                    episode_info1[key].append(info[1][key])
                # learn
                if self.Memory.current_size >= 10*self.args.batch_size:
                    # if self.Memory.current_size >= self.Memory.capacity/100:
                    noise_decrease = True
                    change_lr_flag = True
                    # shared samples
                    tree_index, transitions, ISWeights = self.Memory.sample(self.args.batch_size)
                    s0, a0, r0, s_0, s1, a1, r1, s_1 = self.seperate_transitions(transitions)
                    td_error_ups = []
                    for agent_id, agent in enumerate(self.agents):
                        # unshared samples
                        # tree_index, transitions, ISWeights = self.Memory.sample(self.args.batch_size)
                        # s0, a0, r0, s_0, s1, a1, r1, s_1 = self.seperate_transitions(transitions)
                        other_agents = self.agents.copy()       # list
                        other_agents.remove(agent)
                        td_error_up = agent.learn(s0, a0, r0, s_0, s1, a1, r1, s_1, other_agents, ISWeights)  # train
                        td_error_ups.append(td_error_up)
                        # self.Memory.batch_update(tree_index, td_error_up)
                        # loss data
                        c_loss_one_episode[agent_id].append(agent.policy.c_loss)
                        a_loss_one_episode[agent_id].append(agent.policy.a_loss)
                        if episode_step+1 == self.episode_step:
                            # print('\n'+'---save model at episode %d---'%episode)
                            agent.save_model_net(episode)
                    td_error_up_mean = np.mean(td_error_ups, axis=0)
                    self.Memory.batch_update(tree_index, td_error_up_mean)
                # save data in .mat
                if episode_step+1 == self.episode_step:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    episode_info1.update(episode_info0)
                    colli_times = sum(episode_info0['collision'])
                    f_travel = info[0]['follow_x'] / 1000   # f_travel = episode_info0['follow_x'][-1]
                    feul_cost = info[1]['fuel_consumption']  # in L
                    feul_cost = 100*feul_cost/f_travel
                    episode_info1.update({'colli_times': int(colli_times), 'fuel_100km': feul_cost})
                    scio.savemat(datadir, mdict=episode_info1)
                episode_step += 1
            # change lr
            if change_lr_flag:
                for agent in self.agents:
                    idd_a, idd_c, la, lc = agent.lr_scheduler(episode)
                    lra[idd_a].append(la)
                    lrc[idd_c].append(lc)
            # print
            l_travel = info[0]['leading_x']/1000
            soc = info[1]['SOC']
            soh = info[1]['SOH']
            print('\n'+'episode %d: l_travel %.3fkm, f_travel %.3fkm, collision %d, fuel/100km %.3fL, soc %.3f, soh %.6f'
                  % (episode, l_travel, f_travel, colli_times, feul_cost, soc, soh))
            fuel.append(feul_cost)
            # save loss data
            for i in range(2):
                ck = 'c_loss_%d'%i
                ak = 'a_loss_%d'%i
                c_loss_average[ck].append(np.mean(c_loss_one_episode[i]))
                a_loss_average[ak].append(np.mean(a_loss_one_episode[i]))
            a = np.mean(episode_reward)
            b = np.mean(driver_reward)
            c = np.mean(energy_reward)
            average_reward.append(a)
            driver_average_reward.append(b)
            energy_average_reward.append(c)
            print('episode %d: driver_mean_r %.3f, energy_mean_r %.3f, ep_mean_r %.3f' % (episode, b, c, a) + '\n')
        # save data to .mat
        scio.savemat(self.save_path+'/c_loss_average.mat', mdict=c_loss_average)
        scio.savemat(self.save_path+'/a_loss_average.mat', mdict=a_loss_average)
        scio.savemat(self.save_path+'/lr_a.mat', mdict=lra)
        scio.savemat(self.save_path+'/lr_c.mat', mdict=lrc)
        scio.savemat(self.save_path+'/ep_mean_r.mat', mdict={'ep_r': average_reward})
        scio.savemat(self.save_path+'/driver_mean_r.mat', mdict={'dr_r': driver_average_reward})
        scio.savemat(self.save_path+'/energy_mean_r.mat', mdict={'en_r': energy_average_reward})
        scio.savemat(self.save_path+'/fuel_100km.mat', mdict={'fuel': fuel})

        #  print information
        print('buffer counter:', self.Memory.counter)
        print('buffer current size:', self.Memory.current_size)
        print('replay ratio: %.3f'%(self.Memory.counter/self.Memory.current_size)+'\n')
        print('done:', DONE)
        # find maximal
        names = ['ep_mean_r', 'driver_mean_r', 'energy_mean_r']
        max_reward_list = []
        for data in [average_reward, driver_average_reward, energy_average_reward]:
            max_value = max(data)
            max_reward_list.append((data.index(max_value), max_value))
        for i in range(len(names)):
            print('maximal %s is %.3f at episode %d'%(names[i], max_reward_list[i][1], max_reward_list[i][0]+1))
            
        print('*****PER*****')
