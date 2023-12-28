from tqdm import tqdm           # 进度条
import os
import torch
import numpy as np
import scipy.io as scio
# from common.brain import Brain
from common.brain_PER import Brain

class Evaluator:
    def __init__(self, args, env):
        self.args = args
        self.eva_episode = args.evaluate_episode
        self.episode_step = args.episode_steps
        self.env = env
        self.agents = self._init_agents()
        
        self.save_path = self.args.eva_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Brain(i, self.args, load_or_not=True)
            agents.append(agent)
        return agents

    def evaluate(self):
        average_reward = []  # average_reward of each episode
        driver_average_reward = []
        energy_average_reward = []
        DONE = {}
        # c_loss_average = {'c_loss_0': [], 'c_loss_1': []}
        # a_loss_average = {'a_loss_0': [], 'a_loss_1': []}
        # lra = {'lra_0': [], 'lra_1': []}
        # lrc = {'lrc_0': [], 'lrc_1': []}
        fuel = []  # fuel cost of 100 km of each episode
        for episode in tqdm(range(self.eva_episode)):
            state = self.env.reset()  # reset the environment
        
            episode_reward = []  # sum of reward of a single episode
            driver_reward = []
            energy_reward = []
            episode_step = 0
            # c_loss_one_episode = {0: [], 1: []}
            # a_loss_one_episode = {0: [], 1: []}
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
                        # action = np.clip(normal(action, init_noise), -1, 1)
                        all_actions.append(action)
                state_next, reward, done, info = self.env.step(all_actions)
                # self.buffer.store_episode(state, all_actions, reward, state_next)
                if any(done):
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
                # if self.buffer.current_size >= 10*self.args.batch_size:
                #     noise_decrease = True
                #     change_lr_flag = True
                #     for agent_id, agent in enumerate(self.agents):
                #         transitions = self.buffer.sample(self.args.batch_size)
                #         other_agents = self.agents.copy()
                #         other_agents.remove(agent)
                #         agent.learn(transitions, other_agents)  # train
                #         c_loss_one_episode[agent_id].append(agent.policy.c_loss)
                #         a_loss_one_episode[agent_id].append(agent.policy.a_loss)
                #         if episode_step+1 == self.episode_step:
                #             # print('\n'+'---save model at episode %d---'%episode)
                #             agent.save_model_net(episode)
                # save data in .mat
                if episode_step+1 == self.episode_step:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    episode_info1.update(episode_info0)
                    colli_times = sum(episode_info0['collision'])
                    f_travel = info[0]['follow_x']/1000  # f_travel = episode_info0['follow_x'][-1]
                    feul_cost = info[1]['fuel_consumption']  # in L
                    feul_cost = 100*feul_cost/f_travel
                    episode_info1.update({'colli_times': int(colli_times), 'fuel_100km': feul_cost})
                    scio.savemat(datadir, mdict=episode_info1)
                episode_step += 1
 
            # print
            l_travel = info[0]['leading_x']/1000
            soc = info[1]['SOC']
            soh = info[1]['SOH']
            print(
                '\n'+'episode %d: l_travel %.3fkm, f_travel %.3fkm, collision %d, fuel/100km %.3fL, soc %.3f, soh %.6f'
                % (episode, l_travel, f_travel, colli_times, feul_cost, soc, soh))
            fuel.append(feul_cost)
            # save loss data
            # for i in range(2):
            #     ck = 'c_loss_%d'%i
            #     ak = 'a_loss_%d'%i
            #     c_loss_average[ck].append(np.mean(c_loss_one_episode[i]))
            #     a_loss_average[ak].append(np.mean(a_loss_one_episode[i]))
            a = np.mean(episode_reward)
            b = np.mean(driver_reward)
            c = np.mean(energy_reward)
            average_reward.append(a)
            driver_average_reward.append(b)
            energy_average_reward.append(c)
            print('episode %d: driver_mean_r %.3f, energy_mean_r %.3f, ep_mean_r %.3f'%(episode, b, c, a)+'\n')
        # save data to .mat
        # scio.savemat(self.save_path+'/c_loss_average.mat', mdict=c_loss_average)
        # scio.savemat(self.save_path+'/a_loss_average.mat', mdict=a_loss_average)
        # scio.savemat(self.save_path+'/lr_a.mat', mdict=lra)
        # scio.savemat(self.save_path+'/lr_c.mat', mdict=lrc)
        scio.savemat(self.save_path+'/ep_mean_r.mat', mdict={'ep_r': average_reward})
        scio.savemat(self.save_path+'/driver_mean_r.mat', mdict={'dr_r': driver_average_reward})
        scio.savemat(self.save_path+'/energy_mean_r.mat', mdict={'en_r': energy_average_reward})
        scio.savemat(self.save_path+'/fuel_100km.mat', mdict={'fuel': fuel})
    
        #  print information
        print('done:', DONE)
        # find maximal
        names = ['ep_mean_r', 'driver_mean_r', 'energy_mean_r']
        max_reward_list = []
        for data in [average_reward, driver_average_reward, energy_average_reward]:
            max_value = max(data)
            max_reward_list.append((data.index(max_value), max_value))
        for i in range(len(names)):
            print('maximal %s is %.3f at episode %d'%(names[i], max_reward_list[i][1], max_reward_list[i][0]+1))