import numpy as np
from common.utils2 import get_driving_cycle, get_acc_limit
from common.agent_base import BaseAgent
from common.arguments import get_args
from common.SHEV import SHEV_model
from common.Cell import CellModel_2

args = get_args()
SPEED_LIST = get_driving_cycle(cycle_name=args.scenario_name)  # speed of leading car
ACC_LIST = get_acc_limit(SPEED_LIST, output_max_min=False)  # acceleration of leading car
MILEAGE = sum(SPEED_LIST)/1000  # km


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
        self.leading_x_init = np.random.randint(11, 27)  # random value in [11, 27)
        self.follow_x = 0.
        self.car_length = 5  # unit: m
        self.speed_limit = 120/3.6  # unit: m/s, 120km/h
        self.distance = 0.  # distance between two cars
        # self.distance_init = np.random.randint(5, 20)    # random value in [5, 20)
        self.distance_optimal = 0.  # the optimal of distance for following
        self.distance_min = 3.  # minimum distance for safety
        self.distance_max = 10.
        self.acc_max = 2.5  # ±3 m/s2, this is a limit for output action
        self.acc_min = -2.5
        self.acc_limit = max(abs(self.acc_min), abs(self.acc_max))
        self.train_counter = 0  # counter of training times
        
        self.speed_penalty = False
        self.mean_dict = {'spd': 0.0, 'acc': 0.0, 'distance': 0.0, 'leading_speed': 0.0,
                          'leading_acc': 0.0}
        self.std_dict = {'spd': 1.0, 'acc': 1.0, 'distance': 1.0, 'leading_speed': 1.0,
                         'leading_acc': 1.0}
    
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
        self.leading_x = self.leading_x_init
        self.follow_x = 1.0
        self.distance = self.leading_x-self.follow_x-self.car_length
        # self.distance = self.distance_init
        self.relative_speed = self.leading_speed-self.follow_speed
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # mean & std normalization
        # obs[0] = (self.follow_speed-self.mean_dict['spd'])/self.std_dict['spd']
        # obs[1] = (self.follow_acc-self.mean_dict['acc'])/self.std_dict['acc']
        # obs[2] = (self.distance-self.mean_dict['distance'])/self.std_dict['distance']
        # obs[3] = (self.leading_speed-self.mean_dict['leading_speed'])/self.std_dict['leading_speed']
        # obs[4] = (self.leading_acc-self.mean_dict['leading_acc'])/self.std_dict['leading_acc']
        # maximum normalization
        obs[0] = self.follow_speed/self.speed_limit
        obs[1] = self.follow_acc/self.acc_limit
        obs[2] = self.distance/450        # model9\CLTC_P_v7\episode_data: 96.2%的最大值小于480, 98.6%最小值大于-480
        obs[3] = self.leading_speed/self.speed_limit
        obs[4] = self.leading_acc/self.acc_limit
        # for power demand
        np.save(self.driving_data, [self.follow_speed, self.follow_acc, self.follow_x])
        return obs
    
    def execute(self, action):
        action = action[0]  # acceleration of following car
        action *= abs(self.acc_limit)
        if action > self.acc_max:
            action = self.acc_max
   
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
        acc_tmp = (spd_tmp-self.follow_speed)/sim_time  # true acc
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
        self.info = {'spd': self.follow_speed, 'acc': self.follow_acc, 'distance': self.distance,
                     'leading_x': self.leading_x, 'follow_x': self.follow_x}
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # mean & std normalization
        # obs[0] = (self.follow_speed-self.mean_dict['spd'])/self.std_dict['spd']
        # obs[1] = (self.follow_acc-self.mean_dict['acc'])/self.std_dict['acc']
        # obs[2] = (self.distance-self.mean_dict['distance'])/self.std_dict['distance']
        # obs[3] = (self.leading_speed-self.mean_dict['leading_speed'])/self.std_dict['leading_speed']
        # obs[4] = (self.leading_acc-self.mean_dict['leading_acc'])/self.std_dict['leading_acc']
        # maximum normalization
        obs[0] = self.follow_speed/self.speed_limit
        obs[1] = self.follow_acc/self.acc_limit
        obs[2] = self.distance/450       # model10\mix2_v7_g\episode_data: 99.2%的最大值小于450, 99.8%最小值大于-450
        obs[3] = self.leading_speed/self.speed_limit
        obs[4] = self.leading_acc/self.acc_limit
        # for power demand
        np.save(self.driving_data, [self.follow_speed, self.follow_acc, self.follow_x])
        return obs
    
    def get_reward(self):
        # speed
        if self.speed_penalty:
            r_speed = - self.speed_limit  # 120/3.6=33.33
        else:
            r_speed = 0
        # safety, version 2
        collision = 0
        if self.distance <= 0:
            # print('step %d: distance <= 0, distance: %.1fm, lead_x: %.1fm, follow_x: %.1fm, spd: %.1fm/s, acc: %.1f' %
            #       (self.train_counter, self.distance, self.leading_x, self.follow_x, self.follow_speed, self.follow_acc))
            # r_safe = - self.speed_limit-3*self.follow_acc-self.follow_speed + self.distance
            r_safe = - self.speed_limit - self.follow_acc + self.distance
            collision = 1
        elif 0 < self.distance < self.distance_min:
            r_safe = - self.follow_speed
        elif self.distance > self.distance_max:
            r_safe = -abs(self.distance-self.distance_max)/4
        else:
            r_safe = 1      # 阶梯高度小一点
        TTC = - self.distance/(self.relative_speed+0.001)  # time to collision
        if TTC < 0:
            TTC = -0.1
        if TTC > 10:
            TTC = 10
        # comfort
        r_jerk = - abs(self.jerk)/(self.acc_max-self.acc_min)  # jeck: [-1, 0]
        # total reward
        if r_safe > 0:
            # r_jerk *= 1
            r_jerk *= 2     # v9
        # else:
        #     r_jerk *= 0.1
        r_speed *= 0
        reward = r_speed+r_safe+r_jerk
        self.info.update({'collision': collision, 'TTC': TTC, 'r_jerk': r_jerk, 'r_speed': r_speed,
                          'jerk_value': self.jerk, 'r_safe': r_safe, 'driver_reward': reward})
        return float(reward)

class Energy(BaseAgent):
    def __init__(self):
        super(Energy, self).__init__()
        self.number = '1'
        self.name = 'energy'
        self.EGS = SHEV_model()
        self.cell = CellModel_2()
        self.obs_num = 4  # [soc, tep_a, I_batt, P_mot, follow_x]
        self.action_num = 1  # [eng_power] + noise, [0, 1]; new: [P_batt], [-1,1]
        # parameters for EGS
        self.fuel_cost = 0
        self.fuel_cost_total = 0
        self.P_mot = 0  # unit W
        # parameters for cell
        self.max_power = 150000  # battery package
        self.battery_power_penalty = 0
        # self.SOC_origin = 0.6  # origin value
        self.SOC_origin = np.random.randint(45, 71)/100  # random value in [0.45, 0.71)
        self.SOC_final = 0.2
        self.SOC = self.SOC_origin
        self.SOH = 1.0
        self.dsoh = 0
        self.Tep_a = 25     # 摄氏度
        self.I_batt = 0
        self.OCV_initial = self.cell.ocv_func(self.SOC_origin)*13.87/168
        self.cell_paras = [self.SOC, self.SOH, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.charge_sustain = True
        self.spd, self.acc, self.follow_x = 0, 0, 0          # follow_x: km
    
    def reset_obs(self):
        # EGS
        self.fuel_cost_total = 0
        self.P_mot = 0  # unit W
        # battery
        self.SOC = self.SOC_origin
        self.SOH = 1.0
        self.Tep_a = 25
        self.I_batt = 0
        self.cell_paras = [self.SOC, self.SOH, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.spd, self.acc, self.follow_x = self.communicate()
        # observation
        obs = np.zeros(self.obs_num, dtype=np.float32)
        obs[0] = self.SOC
        obs[1] = (self.Tep_a-25)/(60-25)
        obs[2] = self.I_batt/50
        obs[3] = self.P_mot/150000      # [-1, 1]
        # obs[4] = self.follow_x/MILEAGE
        return obs
    
    def execute(self, action):
        # power demand
        self.spd, self.acc, self.follow_x = self.communicate()
        T_axle, W_axle = self.EGS.T_W_axle(self.spd, self.acc)
        self.P_mot, T_mot, W_mot, Mot_eff = self.EGS.motor_power(T_axle, W_axle)     # unit W
        # action
        # batt_power = action[0]*150000  # unit W
        # isg_power = P_mot - batt_power
        # eng_power = isg_power / 0.98    # easy way
        eng_power = abs(action[0])*62000  # unit W
        # engine & generator
        self.fuel_cost, P_ISG, self.info = self.EGS.run_EGS_2(eng_power, T_axle)
        self.info.update({'W_mot': W_mot, 'T_mot': T_mot, 'P_mot': self.P_mot/1000,
                          'Mot_eff': Mot_eff, 'T_axle': T_axle, 'W_axle': W_axle})
        self.fuel_cost_total += self.fuel_cost
        self.info.update({'fuel_consumption': self.fuel_cost_total})
        # battery
        P_batt = self.P_mot-P_ISG        # unit W
        # battery power limit and penalty
        if P_batt > self.max_power:
            P_batt = self.max_power
        if P_batt < -self.max_power:
            P_batt = -self.max_power
        if abs(P_batt) >= 90000:
            self.battery_power_penalty = (abs(P_batt)-90000)/1000/60        # [0, 1]
        else:
            self.battery_power_penalty = 0
        # update parameters
        self.cell_paras, self.dsoh, self.I_batt, self.done, info_batt = \
            self.cell.run_cell(P_batt, self.cell_paras)
        self.SOC = self.cell_paras[0]
        self.Tep_a = self.cell_paras[4]
        self.info.update(info_batt)
        # observation
        obs = np.zeros(self.obs_num, dtype=np.float32)
        obs[0] = self.SOC
        obs[1] = (self.Tep_a-25)/(60-25)
        obs[2] = self.I_batt/50
        obs[3] = self.P_mot/150000
        # obs[4] = self.follow_x/MILEAGE
        return obs
    
    def get_reward(self):
        # EGS
        w1_egs = 8.46  # #92 Fuel price: ￥9.01 per L1
        fuel_cost = w1_egs*self.fuel_cost
        self.info.update({'fuel_cost': fuel_cost})
        # battery
        # reward about SoC
        if self.charge_sustain:  # keep SoC stable near a low level, 0.3
            delta_SoC = self.SOC-self.SOC_final
            if self.SOC >= 0.80:
                w1 = 20
                # w1 = 50     # v15
            elif self.SOC <= 0.15:
                w1 = 20
                # w1 = 50  # v15
            else:
                w1 = 2.0
                # w1 = 5  # v15
        else:  # the SoC is expected to decrease evenly
            lamda = (self.SOC_origin-self.SOC_final)/MILEAGE  # 除以本次行程的总里程数, km
            SoC_r = self.SOC_origin-lamda*self.follow_x      # follow_x should be in state space
            delta_SoC = self.SOC-SoC_r
            w1 = 10
            # if self.SOC > 0.90 or self.SOC < 0.10:
            #     w1 *= 5
        cost_SOC = w1*abs(delta_SoC)
        # reward about SOH
        w2 = 13000       # SoH reward coefficient    ￥13042.5 for replacing a new battery
        cost_SOH = w2*self.dsoh
        # reward about P_batt, too big is not good
        w3 = 1      # self.battery_power_penalty: [0, 1],  3-->1
        cost_power = w3 * self.battery_power_penalty        # [0, w3]
        # total cost and reward
        bttery_reward = -(cost_SOC+cost_SOH)        # [-1, 0]
        reward = -fuel_cost+bttery_reward-cost_power
        self.info.update({'battery_reward': bttery_reward, 'delta_SOC': delta_SoC,
                          'cost_SOC': cost_SOC, 'cost_SOH': cost_SOH, 'cost_power': cost_power,
                          'energy_reward': reward})
        return reward
    
    def communicate(self):
        x = np.load(self.driving_data)
        spd = x[0]
        acc = x[1]
        follow_x = x[2]     # m
        follow_x /= 1000  # km
        return spd, acc, follow_x
