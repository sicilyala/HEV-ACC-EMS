from tqdm import tqdm  # 进度条
import os
import torch
import numpy as np
from numpy.random import normal  # normal distribution
from brain import Brain
from replay_buffer import Buffer

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.buffer = Buffer(args)
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
    
    def run(self):
        """DDPG style"""
        average_reward = []  # average_reward of each episode
        driver_average_reward = []
        EGS_average_reward = []
        battery_average_reward = []
        step_reward_of_one_episode = []
        # all_noise_rate = []
        info_episode = {}
        # info_episode_1 = {}
        # info_episode_2 = {}
        # info_episode_3 = {}
        noise_decrease = False
        noise_rate = self.args.noise_rate
        DONE = []
        c_loss_average = {0: [], 1: [], 2: []}
        a_loss_average = {0: [], 1: [], 2: []}
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            episode_reward_sum = []  # sum of reward of a single episode
            driver_reward = []
            EGS_reward = []
            battery_reward = []
            episode_step = 0
            if noise_decrease:
                noise_rate *= self.args.noise_discout_rate
            # noise_rate = self.noise_init * np.exp(-episode/self.episode_num)
            # change_lr_flag = False
            info_episode_step = []  # [[{}, {}, {}], [{}, {}, {}], ...]
            setp_reward = []
            # all_noise_rate.append(noise_rate)
            # running mean & std
            s_dict = {'spd': [], 'acc': [], 'distance': [], 'leading_speed': [],
                      'leading_acc': [],
                      'T_eng': [], 'P_eng': [],
                      'Tep_a': [], 'I_batt': []}
            c_loss_one_episode = {0: [], 1: [], 2: []}
            a_loss_one_episode = {0: [], 1: [], 2: []}
            info = []
            while episode_step < self.episode_step:
                # running mean & std
                ob = []
                for i in range(len(state)):
                    for j in range(len(state[i])):
                        ob.append(state[i][j])
                for key, value in enumerate(s_dict):
                    s_dict[value].append(ob[key])
                
                all_actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action2(state[agent_id])
                        action_i = np.clip(normal(action, noise_rate), -1, 1)
                        all_actions.append(action_i)
                state_next, reward, done, info = self.env.step(all_actions)
                self.buffer.store_episode(state, all_actions, reward, state_next)
                # state_next, reward, done, info, accs_true = self.env.step(all_actions)
                # self.buffer.store_episode(state, accs_true, reward, state_next)
                if any(done):
                    print('failure in step %d of episode %d'%(episode_step, episode))
                    DONE.append((episode_step, episode))
                    break
                state = state_next
                episode_reward_sum.append(reward)  # reward: [r1, r2, r3]
                driver_reward.append(reward[0])
                EGS_reward.append(reward[1])
                battery_reward.append(reward[2])
                setp_reward.append(np.mean(reward))
                info_episode_step.append(info)  # info:[{}, {}, {}], info_episode_step: [[{}, {}, {}], [{}, {}, {}], ...]
                # learn
                if self.buffer.current_size >= self.args.batch_size:
                    noise_decrease = True
                    # change_lr_flag = True
                    for agent_id, agent in enumerate(self.agents):
                        transitions = self.buffer.sample(self.args.batch_size)
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)  # train
                        c_loss_one_episode[agent_id].append(agent.policy.c_loss)
                        a_loss_one_episode[agent_id].append(agent.policy.a_loss)
                        if episode in self.args.save_episodes:
                            if episode_step+1 == self.episode_step:
                                print('\n'+'---save model at episode %d---'%episode)
                                agent.save_model_net(episode)
                episode_step += 1
            # print
            l_travel = info[0]['leading_x']/1000
            f_travel = info[0]['follow_x']/1000
            feul_cost = info[1]['fuel_consumption']  # in L
            feul_cost = 100*feul_cost/f_travel
            soc = info[2]['SoC']
            print('episode%d:l_travel %.3fkm, f_travel %.3fkm, fuel/100km %.3fL, soc %.3f' %
                  (episode, l_travel, f_travel, feul_cost, soc))
            # save loss data
            for i in range(3):
                c_loss_average[i].append(np.mean(c_loss_one_episode[i]))
                a_loss_average[i].append(np.mean(a_loss_one_episode[i]))
            average_reward.append(np.mean(episode_reward_sum))
            driver_average_reward.append(np.mean(driver_reward))
            EGS_average_reward.append(np.mean(EGS_reward))
            battery_average_reward.append(np.mean(battery_reward))
            step_reward_of_one_episode.append(setp_reward)
            # if (self.episode_num-1-episode) < 300:  # only save last 300 episodes
            #     info_episode.update({episode: info_episode_step})
            np.save(self.save_path_episode+'/data_ep%d' % episode, info_episode_step)
            '''
            # save in batch, when episode_num==599
            # if 200 <= (self.episode_num-episode) < 300:  # [300, 399]
            #     info_episode_1.update({episode: info_episode_step})
            # if 100 <= (self.episode_num-episode) < 200:  # [400, 499]
            #     info_episode_2.update({episode: info_episode_step})
            # if 0 <= (self.episode_num-episode) < 100:  # [500, 599]
            #     info_episode_3.update({episode: info_episode_step})
            '''
            # running mean & std of observation
            self._update_mean_std(0, ['spd', 'acc', 'distance', 'leading_speed',
                                      'leading_acc'], s_dict)
            self._update_mean_std(1, ['T_eng', 'P_eng'], s_dict)
            self._update_mean_std(2, ['Tep_a', 'I_batt'], s_dict)
            # std of reward for scaling
            reward_scale = False
            if reward_scale:
                reward_list = [driver_reward, EGS_reward, battery_reward]
                for i in range(3):
                    # # r_std_tmp = np.std(reward_list[i])*self.args.miu + self.env.agents[i].reward_std*(1-self.args.miu)
                    # # self.env.agents[i].reward_std = r_std_tmp
                    miu = 0.9
                    r_std_origin = self.env.agents[i].reward_std
                    # # r_std_tmp = abs(r_std_origin*np.min(reward_list[i]))*miu+r_std_origin*(1-miu)   # poor performance!
                    r_std_tmp = abs(np.min(reward_list[i]))*miu+r_std_origin*(1-miu)
                    self.env.agents[i].reward_std = r_std_tmp
            # print(self.env.agents[i].reward_std)
            # lr scheduler              !cause training faliure of battery!
            # if change_lr_flag:
            #     for agent in self.agents:
            #         agent.lr_scheduler()
        np.save(self.save_path+'/average_reward', average_reward)
        np.save(self.save_path+'/driver_average_reward', driver_average_reward)
        np.save(self.save_path+'/EGS_average_reward', EGS_average_reward)
        np.save(self.save_path+'/battery_average_reward', battery_average_reward)
        np.save(self.save_path+'/average_reward_of_one_episode', step_reward_of_one_episode)
        # np.save(self.save_path+'/info_episode', info_episode)
        # np.save(self.save_path+'/info_episode_1', info_episode_1)
        # np.save(self.save_path+'/info_episode_2', info_episode_2)
        # np.save(self.save_path+'/info_episode_3', info_episode_3)
        # np.save(self.save_path + '/noise_rate', all_noise_rate)
        np.save(self.save_path+'/c_loss_average', c_loss_average)
        np.save(self.save_path+'/a_loss_average', a_loss_average)
        
        print('buffer counter:', self.buffer.counter)
        print('buffer current size:', self.buffer.current_size)
        print('replay ratio: %.3f'%(self.buffer.counter/self.buffer.current_size))
        print('done:', DONE)