import numpy as np
from utils2 import get_driving_cycle, get_acc_limit
from agent_base import BaseAgent
from arguments import get_args
from SHEV import SHEV_model
from Cell import CellModel_1, CellModel_2

args = get_args()
SPEED_LIST = get_driving_cycle(cycle_name=args.scenario_name)  # speed of leading car
# MAX_STEP = len(SPEED_LIST)
ACC_LIST = get_acc_limit(SPEED_LIST, output_max_min=False)       # acceleration of leading car

class Driver(BaseAgent):
    def __init__(self):
        super(Driver, self).__init__()
        self.number = '0'
        self.name = 'driver'
        self.obs_num = 5
        # observation: np.array([follow_speed, follow_acc, distance, leading_speed, leading_speed1,leading_speed2])
        # observation: np.array([follow_speed, follow_acc, distance, leading_speed, leading_acc])
        self.action_num = 1  # action: [follow_acc]: [-1, 1]+noise
        self.leading_speed = 0.  # speed of leading car      m/s
        # self.leading_speed_1 = 0.
        # self.leading_speed_2 = 0.
        self.follow_speed = 0.  # speed of following car
        self.relative_speed = 0.
        self.leading_acc = 0.  # acceleration of leading car   m/s2
        self.follow_acc = 0.  # output by actor network
        self.jerk = 0.
        self.leading_x = 0.  # how far the leading car has traveled   m
        self.follow_x = 0.
        self.car_length = 5  # unit: m
        self.speed_limit = 120/3.6  # unit: m/s, 120km/h
        self.distance = 0.  # distance between two cars
        self.distance_optimal = 0.  # the optimal of distance for following
        self.distance_min = 3.  # minimum distance for safety
        self.distance_max = 10.
        # self.speed_list = speed_list  # speed of leading car
        # self.acc_list = acc_list    # acceleration of leading car
        self.acc_max = 3.0  # ±3 m/s2, this is a limit for output action
        self.acc_min = -3.5
        self.acc_limit = max(abs(self.acc_min), abs(self.acc_max))
        # self.acc_max, self.acc_min = get_acc_limit(self.speed_list)  # limit for output acceleration
        self.train_counter = 0  # counter of training times
        
        # self.speed_optimal = 45/3.6
        self.speed_penalty = False
        self.mean_dict = {'spd': 0.0, 'acc': 0.0, 'distance': 0.0, 'leading_speed': 0.0,
                          'leading_acc': 0.0}
        self.std_dict = {'spd': 1.0, 'acc': 1.0, 'distance': 1.0, 'leading_speed': 1.0,
                         'leading_acc': 1.0}
        self.reward_std = 1
    
    def reset_obs(self):
        """reset observation"""
        self.train_counter = 0
        self.jerk = 0
        self.speed_penalty = False
        self.distance_min = 3.
        self.distance_max = 10.
        self.follow_speed = 0.
        self.follow_acc = 0.
        # self.leading_speed = self.speed_list[0]
        # self.leading_acc = self.acc_list[0]
        self.leading_speed = 0.
        self.leading_acc = 0.
        # self.leading_speed_1 = 0.
        # self.leading_speed_2 = 0.
        # self.leading_acc = 0.
        self.leading_x = 20.
        self.follow_x = 0.
        self.distance = self.leading_x-self.follow_x-self.car_length
        self.relative_speed = self.leading_speed-self.follow_speed
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.follow_speed-self.mean_dict['spd'])/self.std_dict['spd']
        obs[1] = (self.follow_acc-self.mean_dict['acc'])/self.std_dict['acc']
        obs[2] = (self.distance-self.mean_dict['distance'])/self.std_dict['distance']
        obs[3] = (self.leading_speed-self.mean_dict['leading_speed'])/self.std_dict['leading_speed']
        obs[4] = (self.leading_acc-self.mean_dict['leading_acc'])/self.std_dict['leading_acc']
        # obs[5] = (self.leading_speed_2-self.mean_dict['leading_speed_2'])/self.std_dict['leading_speed_2']
        # maximum normalization
        # obs[0] = self.follow_speed / self.speed_limit
        # obs[1] = self.follow_acc / self.acc_max
        # obs[2] = self.distance / 100
        # obs[3] = self.leading_speed / self.speed_limit
        # obs[4] = self.leading_speed_1 / self.speed_limit
        # obs[5] = self.leading_speed_2 / self.speed_limit
        #
        np.save(self.driving_data, [self.follow_speed, self.follow_acc, self.follow_x])
        return obs
    
    def execute(self, action):
        action = action[0]  # acceleration of following car
        action *= abs(self.acc_limit)
        if action > self.acc_max:
            action = self.acc_max
        # embed rules   try TTC
        # if (self.distance < 5) and (action > -1.50):
        # if (self.distance < 10) and (action > -2.50):
        #     action = -2.50
        sim_time = 1  # simulation time interval
        # actual speed and acc considering speed limit
        spd_tmp = self.follow_speed+action*sim_time
        # if spd_tmp < 0 or spd_tmp > self.speed_limit:
        if spd_tmp > self.speed_limit:
            self.speed_penalty = True
        else:
            self.speed_penalty = False
        if spd_tmp < 0:
            spd_tmp = 0
        if spd_tmp > self.speed_limit:
            spd_tmp = self.speed_limit
        acc_tmp = (spd_tmp-self.follow_speed)/sim_time      # true acc
        action = acc_tmp
        # update distance
        self.leading_x += (sim_time*self.leading_speed)
        # self.leading_x += (self.leading_speed*sim_time+0.5*self.leading_acc*pow(sim_time, 2))
        #  more accurate way of calculation?   done    actually same
        self.follow_x += (self.follow_speed*sim_time+0.5*action*pow(sim_time, 2))
        self.distance = self.leading_x-self.follow_x-self.car_length
        # update speed, acceleration, jerk
        self.follow_speed = spd_tmp
        self.leading_speed = SPEED_LIST[self.train_counter]
        self.leading_acc = ACC_LIST[self.train_counter]
        # if self.train_counter > MAX_STEP-3:
        #     self.leading_speed_1 = 0.
        #     self.leading_speed_2 = 0.
        # else:
        #     self.leading_speed_1 = SPEED_LIST[self.train_counter+1]
        #     self.leading_speed_2 = SPEED_LIST[self.train_counter+2]
        
        self.train_counter += 1
        self.relative_speed = self.leading_speed-self.follow_speed
        self.jerk = action-self.follow_acc
        self.follow_acc = action
        # optimal and safe distance
        # self.distance_optimal = 15.336+95.9*np.arctanh(0.02*self.leading_speed-0.008)
        self.distance_min = 3+self.follow_speed*0.8+pow(self.follow_speed, 2)/(7.5*2)
        self.distance_max = 10+self.follow_speed+0.0825*(self.follow_speed**2)
        if self.distance_max > 160:
            self.distance_max = 160
        self.info = {'speed': self.follow_speed, 'acceleration': self.follow_acc, 'distance': self.distance,
                     'leading_x': self.leading_x, 'follow_x': self.follow_x}
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.follow_speed-self.mean_dict['spd'])/self.std_dict['spd']
        obs[1] = (self.follow_acc-self.mean_dict['acc'])/self.std_dict['acc']
        obs[2] = (self.distance-self.mean_dict['distance'])/self.std_dict['distance']
        obs[3] = (self.leading_speed-self.mean_dict['leading_speed'])/self.std_dict['leading_speed']
        obs[4] = (self.leading_acc-self.mean_dict['leading_acc'])/self.std_dict['leading_acc']
        # obs[5] = (self.leading_speed_2-self.mean_dict['leading_speed_2'])/self.std_dict['leading_speed_2']
        # maximum normalization
        # obs[0] = self.follow_speed/self.speed_limit
        # obs[1] = self.follow_acc/self.acc_max
        # obs[2] = self.distance/100
        # obs[3] = self.leading_speed/self.speed_limit
        # obs[4] = self.leading_speed_1/self.speed_limit
        # obs[5] = self.leading_speed_2/self.speed_limit
        #
        np.save(self.driving_data, [self.follow_speed, self.follow_acc, self.follow_x])
        return obs, action/abs(self.acc_limit)
    
    def get_reward(self):
        # speed
        if self.speed_penalty:
            r_speed = - self.speed_limit  # 120/3.6=33.33
        else:
            r_speed = 0
        # safety, version 1
        # if self.distance <= 0.0:
        #     # if self.distance <= self.distance_min:
        #     print('step %d: distance <= safe distance %.1fm!'%(self.train_counter, 0))
        #     collision = 1
        #     r_ttc = -200  # collision penalty, np.log(0.001/4)=-8.29, seen as the minimum value
        #     # self.done = True
        # else:
        #     collision = 0
        #     ttc = - self.distance/(self.relative_speed+0.001)  # time to collision
        #     if 0 < ttc <= 4:
        #         r_ttc = np.log(ttc/4)
        #     elif ttc < 0:  # 当ttc为负时，表示永远追不上，惩罚
        #         # print('step %d: TTC < 0!'%self.train_counter)
        #         r_ttc = -8.0
        #     else:  # 当ttc>4时，惩罚它越来越大的情况
        #         # print('step %d: TTC > 4!'%self.train_counter)
        #         # r_ttc = -10 - (ttc-4)   # 当相对速度非常小时，可能出现极端惩罚值，-10000
        #         r_ttc = 0.0
        # safety, version 2
        collision = 0
        r_distance_to_EGS = 0
        if self.distance <= 0:
            print('step %d: distance <= 0, distance: %.1fm, lead_x: %.1fm, follow_x: %.1fm, spd: %.1fm/s, acc: %.1f' %
                  (self.train_counter, self.distance, self.leading_x, self.follow_x, self.follow_speed, self.follow_acc))
            r_safe = - self.speed_limit - 3*self.follow_acc - self.follow_speed
            collision = 1
        elif 0 < self.distance < self.distance_min:
            r_safe = - self.follow_speed
        elif self.distance > self.distance_max:
            r_safe = -abs(self.distance-self.distance_max)/4
            r_distance_to_EGS = r_safe
        else:
            r_safe = 15
        TTC = - self.distance/(self.relative_speed+0.001)  # time to collision
        #
        np.save("D:/SEU2/Program1/MADDPG-program/common/data/r_distance_to_EGS.npy", r_distance_to_EGS)
        '''
        # efficiency
        # hdw = (self.distance+self.car_length) / (self.follow_speed+0.001)   # headway time
        # if hdw <= 0:
        #     # print('step %d: headway time <= 0!' % self.train_counter)
        #     r_hdw = -10  #
        # elif hdw >= 4:
        #     # print('step %d: headway time >= 4!' % self.train_counter)
        #     r_hdw = -10 - (hdw-4)
        # else:    # 正态分布公式    mu = 0.4226     sigma = 0.4365
        #     r_hdw = (np.exp(-(np.log(hdw) - 0.4226) ** 2 / (2 * 0.4365 ** 2)) / (hdw * 0.4365 * np.sqrt(2 * np.pi)))
        '''
        # comfort
        # base_value = pow((self.acc_max-self.acc_min)/1, 2)  # sample_interval = 1, base_value==36
        # r_jerk = -pow(self.jerk, 2)/base_value  # jerk: defined as the change rate of acceleration [0, 1]
        r_jerk = - abs(self.jerk)/(self.acc_max-self.acc_min)  # jeck: [-1, 1]
        # optimal distance
        # distance_diffrence = self.distance-self.distance_optimal
        # total reward
        # r_od = -abs(distance_diffrence) * 0.9
        # r_jerk *= 150       # 165
        # reward = r_ttc + r_od + r_jerk + r_speed
        # version 2
        # r_safe *= 5
        # r_jerk *= 5
        r_speed *= 0
        reward = r_speed+r_safe+r_jerk
        scaled_reward = reward/self.reward_std
        # self.info.update({'collision': collision, 'distance_difference': distance_diffrence,
        #                   'r_ttc': r_ttc, 'r_od': r_od, 'r_jerk': r_jerk, 'r_speed': r_speed,
        #                   'driver_reward': reward, 'driver_scaled_reward': scaled_reward})
        self.info.update({'collision': collision, 'TTC': TTC, 'r_jerk': r_jerk, 'r_speed': r_speed,
                          'jerk_value': self.jerk, 'r_safe': r_safe,
                          'driver_reward': reward, 'driver_scaled_reward': scaled_reward})
        return float(scaled_reward)

class EngineGeneratorSet(BaseAgent):
    def __init__(self):
        super(EngineGeneratorSet, self).__init__()
        self.number = '1'
        self.name = 'engine-generator set'
        self.obs_num = 2  # [spd, acc], new: [W_eng, T_eng, power_difference], new: [T_eng, P_eng_by_batt,]
        self.action_num = 1  # [power_eng, alpha]: [-1, 1]+noise, new: [W_eng]
        self.T_eng = 0
        self.P_eng = 0
        self.P_ISG = 0
        self.Gen_eff = 1.0
        self.Eng_fuel_eff = 0.2
        # self.T_axle = 0
        self.fuel_cost = 0
        self.fuel_cost_total = 0
        self.mean_dict = {'T_eng': 0.0, 'P_eng': 0.0}
        self.std_dict = {'T_eng': 1.0, 'P_eng': 1.0}
        self.reward_std = 1
        self.SHEV = SHEV_model()
    
    def reset_obs(self):
        self.fuel_cost_total = 0
        self.T_eng = 0
        self.P_eng = 0
        self.P_ISG = 0
        self.Gen_eff = 1.0
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.T_eng-self.mean_dict['T_eng'])/self.std_dict['T_eng']
        obs[1] = (self.P_eng-self.mean_dict['P_eng'])/self.std_dict['P_eng']
        # maxi
        # obs[0] = self.T_eng / 306
        # obs[1] = self.P_eng / 62000
        return obs
    
    def calculate_power(self):
        spd, acc = self.communicate()
        T_axle, W_axle = self.SHEV.T_W_axle(spd, acc)
        P_mot, T_mot, W_mot, Mot_eff = self.SHEV.motor_power(T_axle, W_axle)
        # T_eng, W_eng = self.SHEV.T_W_engine(eng_power, self.T_axle)
        # P_ISG, T_ISG, W_ISG, Gen_eff, T_eng = self.SHEV.ISG_power(T_eng, W_eng)
        # self.info.update({'motor_speed': W_mot, 'motor_torque': T_mot,
        #                   'motor_power': P_mot/1000, 'Mot_eff': Mot_eff})
        self.info = {'motor_speed': W_mot, 'motor_torque': T_mot,
                     'motor_power': P_mot/1000, 'Mot_eff': Mot_eff}
        return P_mot, T_axle, W_axle
    
    def execute(self, action):
        if self.P_eng > 62000:  # TODO Inflict punishment?
            self.P_eng = 62000
        W_eng = 377*abs(action[0])
        self.T_eng = self.P_eng/W_eng
        self.P_eng, self.T_eng, W_eng, self.Gen_eff, self.Eng_fuel_eff, self.fuel_cost, self.P_ISG, infoo = \
            self.SHEV.run_EGS(self.P_eng, self.T_eng, W_eng)
        # 此处无需返回 W_eng
        self.info.update(infoo)
        self.fuel_cost_total += self.fuel_cost
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.T_eng-self.mean_dict['T_eng'])/self.std_dict['T_eng']
        obs[1] = (self.P_eng-self.mean_dict['P_eng'])/self.std_dict['P_eng']
        # maxi
        # obs[0] = self.T_eng/306
        # obs[1] = self.P_eng/62000
        return obs, W_eng/377
    
    def get_reward(self):
        #
        r_distance_to_EGS = np.load("D:/SEU2/Program1/MADDPG-program/common/data/r_distance_to_EGS.npy")
        #
        w1 = 8.46  # #92 Fuel price: ￥9.01 per L
        w2 = 1
        reward = -w1*self.fuel_cost+w2*(self.Eng_fuel_eff-0.5) + r_distance_to_EGS/100
        scaled_reward = reward/self.reward_std
        self.info.update({'EGS_reward': reward, 'fuel_consumption': self.fuel_cost_total,
                          'EGS_scaled_reward': scaled_reward})
        return float(scaled_reward)
    
    def communicate(self):
        x = np.load(self.driving_data)
        spd = x[0]
        acc = x[1]
        return spd, acc

class Battery(BaseAgent):
    def __init__(self):
        super(Battery, self).__init__()
        self.number = '2'
        self.name = 'battery'
        self.obs_num = 3  # [Tep_a, I_batt, SoC]
        self.action_num = 1  # [battery_power]: [-1, 1]+noise
        self.max_power = 150000
        self.battery_power_penalty = 0
        self.SOC_origin = 0.60  # origin value
        self.SOC = self.SOC_origin
        self.SOH = 1.0
        self.aging_cost_total = 0
        self.life_depleted_fraction = 0
        self.Tep_c = 25  # 摄氏度
        self.Tep_s = 25
        self.Tep_a = 25
        self.I_batt = 0
        self.mean_dict = {'Tep_a': 0.0, 'I_batt': 0.0}
        self.std_dict = {'Tep_a': 1.0, 'I_batt': 1.0}
        self.reward_std = 1
        self.charge_sustain = True
        self.paras_list = [self.SOC, self.SOH, self.Tep_c, self.Tep_s, self.Tep_a, 0, 0, 0]
        self.cell_1 = CellModel_1()  # Equivalent circuit model
        self.cell_2 = CellModel_2()  # Electro-thermal-aging model
    
    def _reset_attribute(self):
        self.battery_power_penalty = 0
        self.aging_cost_total = 0
        self.SOC = self.SOC_origin
        self.SOH = 1.0
        self.Tep_c = 25  # 摄氏度
        self.Tep_s = 25
        self.Tep_a = 25
        self.cell_1.accumulated_Ah = 0
        self.I_batt = 0
        self.paras_list = [self.SOC, self.SOH, self.Tep_c, self.Tep_s, self.Tep_a, 0, 0, 0]
    
    def reset_obs(self):
        self._reset_attribute()
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.Tep_a-self.mean_dict['Tep_a'])/self.std_dict['Tep_a']
        obs[1] = (self.I_batt-self.mean_dict['I_batt'])/self.std_dict['I_batt']
        obs[2] = self.SOC
        # maxi
        # obs[0] = self.Tep_a/55
        # obs[1] = self.I_batt/40
        # obs[2] = self.SOC
        return obs
    
    def execute(self, battery_power):
        if battery_power > self.max_power:
            battery_power = self.max_power
        if battery_power < -self.max_power:
            battery_power = -self.max_power
        if abs(battery_power) > 90000:
            self.battery_power_penalty = (abs(battery_power)-90000)/1000
        else:
            self.battery_power_penalty = 0
        
        cellmodel = 2
        if cellmodel == 1:
            # Equivalent circuit model
            self.SOC, self.Tep_c, self.Tep_s, self.Tep_a, self.life_depleted_fraction, \
            self.I_batt, self.done, self.info = \
                self.cell_1.run_cell(battery_power, self.SOC, self.Tep_c, self.Tep_s)
        else:
            # electro-thermal-aging model
            self.paras_list, self.life_depleted_fraction, self.I_batt, self.done, self.info = \
                self.cell_2.run_cell(battery_power, self.paras_list)
            self.SOC = self.paras_list[0]
            self.Tep_a = self.paras_list[4]
        
        self.aging_cost_total += self.life_depleted_fraction
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = (self.Tep_a-self.mean_dict['Tep_a'])/self.std_dict['Tep_a']
        obs[1] = (self.I_batt-self.mean_dict['I_batt'])/self.std_dict['I_batt']
        obs[2] = self.SOC
        # # maxi
        # obs[0] = self.Tep_a/55
        # obs[1] = self.I_batt/40
        # obs[2] = self.SOC
        return obs
    
    def get_reward(self):
        # reward about SoC
        if self.charge_sustain:  # keep SoC stable near a low level, 0.3
            delta_SoC = self.SOC-0.3
            # if self.SOC >= 0.95 or self.SOC <= 0.05:
            #     w1 = 20.0
            # elif 0.85 < self.SOC < 0.95 or 0.05 < self.SOC < 0.15:
            #     w1 = 10.0
            if self.SOC >= 0.90:
                w1 = 20
            elif self.SOC <= 0.10:
                w1 = 20
            else:
                w1 = 1.0
        else:  # the SoC is expected to decrease evenly
            x = np.load(self.driving_data)
            follow_x = x[2]/1000  # unit: km
            lamda = (self.SOC_origin-0.2)/100       # 除以本次行程的总里程数
            SoC_r = self.SOC_origin-lamda*follow_x
            delta_SoC = self.SOC-SoC_r
            w1 = 1  # 100
        
        # w2 = 13042.5*5    # SoH reward coefficient    ￥13042.5 for replacing a new battery
        w2 = 13042.5
        cost_SOC = w1*abs(delta_SoC)
        cost_SOH = w2*self.life_depleted_fraction
        # cost_power = 0.1 * self.battery_power_penalty
        cost_power = 0.0*self.battery_power_penalty
        # self.battery_power_penalty = False
        cost = cost_SOC+cost_SOH+cost_power
        reward = -cost
        scaled_reward = reward/self.reward_std
        self.info.update({'battery_reward': reward, 'delta_SoC': delta_SoC,
                          'cost_SOC': cost_SOC, 'cost_SOH': cost_SOH, 'aging_fraction': self.aging_cost_total,
                          'battery_power_penalty': cost_power, 'battery_scaled_reward': scaled_reward})
        return float(scaled_reward)