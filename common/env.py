from car_agents import Driver, EngineGeneratorSet, Battery

class Env:
    def __init__(self):
        self.agents = self._init_agents()
        self.agent_num = len(self.agents)
    
    @staticmethod
    def _init_agents():
        driver = Driver()
        EGS = EngineGeneratorSet()
        battery = Battery()
        agents = [driver, EGS, battery]
        return agents
       
    def reset(self):
        obs_all = []
        for i in range(self.agent_num):
            observation = self.agents[i].reset_obs()
            obs_all.append(observation)
        return obs_all
    
    def step(self, actions):
        obs_all = []
        reward_all = []
        done_all = []
        info_all = []
        accs_true = []
        # agent 0: Driver
        obs, acc_true = self.agents[0].execute(actions[0])
        obs_all.append(obs)
        accs_true.append([acc_true])
        reward_all.append(self.agents[0].get_reward())
        done_all.append(self.agents[0].get_done())
        info_all.append(self.agents[0].get_info())
        # # agent 1: EGS, agent 2: Battery
        """
        # eng_power = abs(actions[1][0]) * 62000
        # alpha = abs(actions[1][1])   # [-1, 1]       # 先改成正数，降低训练难度
        # batt_power = (actions[2][0]) * self.agents[2].max_power   # [-1, 1]
        # self.agents[1].info.update({'alpha': alpha})
        # P_mot, P_ISG, Gen_eff = self.agents[1].calculate_power(eng_power)
        # P_batt_by_EGS = P_mot - P_ISG       # TODO + or - ??
        # power_difference = P_batt_by_EGS - batt_power   # in W
        # self.agents[1].power_difference = power_difference
        # self.agents[1].info.update({'power_difference': power_difference})
        # allocate_ISG = power_difference * (1-alpha)
        # allocate_eng = allocate_ISG / Gen_eff
        # allocate_battery = power_difference * alpha
        # final_eng_power = eng_power + allocate_eng
        # final_batt_power = batt_power + allocate_battery
        # # execute actions
        # actions_2 = [final_eng_power, final_batt_power]
        """
        P_batt = (actions[2][0]) * self.agents[2].max_power
        P_mot, T_axle, W_axle = self.agents[1].calculate_power()
        P_ISG_need = P_mot - P_batt      # in W
        if (P_ISG_need <= 500) or (T_axle < 0):
        # if P_ISG_need <= 500:
        # if (P_ISG_need <= 500) or (W_axle == 0 and T_axle < 0):
            P_ISG_need = 0
            # P_batt = P_mot
        self.agents[1].P_eng = P_ISG_need / self.agents[1].Gen_eff
        # print(self.agents[1].P_eng)
        # print(self.agents[1].P_ISG)
        # agent EGS
        obs, acc_true2 = self.agents[1].execute(actions[1])
        obs_all.append(obs)
        accs_true.append([acc_true2])
        reward_all.append(self.agents[1].get_reward())
        done_all.append(self.agents[1].get_done())
        info_all.append(self.agents[1].get_info())
        # agent battery
        # print(self.agents[1].P_eng)
        # print(self.agents[1].P_ISG)
        P_batt = P_mot - self.agents[1].P_ISG       # in Watt
        obs_all.append(self.agents[2].execute(P_batt))
        reward_all.append(self.agents[2].get_reward())
        done_all.append(self.agents[2].get_done())
        info_all.append(self.agents[2].get_info())
        accs_true.append([P_batt/self.agents[2].max_power])

        print('Driver reward: %.2f; EGS reward: %.2f; Battery reward: %.2f.' %
              (reward_all[0], reward_all[1], reward_all[2]))
        return obs_all, reward_all, done_all, info_all,
        # return obs_all, reward_all, done_all, info_all, accs_true
    