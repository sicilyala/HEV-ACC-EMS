# 在car_agents_2基础上，为适配DQN做了小改动

import numpy as np
from common.SHEV import SHEV_model
from common.Cell import CellModel_2

class EMS_DQN:
    def __init__(self):
        self.done = False
        self.info = {}
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
        self.SOC_origin = 0.6  # origin value
        # self.SOC_origin = np.random.randint(45, 71)/100  # random value in [0.45, 0.71)
        self.SOC_final = 0.2
        self.SOC = self.SOC_origin
        self.SOH = 1.0
        self.dsoh = 0
        self.Tep_a = 25     # 摄氏度
        self.I_batt = 0
        self.OCV_initial = self.cell.ocv_func(self.SOC_origin)*13.87/168
        self.cell_paras = [self.SOC, self.SOH, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.charge_sustain = True
        # self.spd, self.acc, self.follow_x = 0, 0, 0          # follow_x: km
    
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
        # observation
        obs = np.zeros(self.obs_num, dtype=np.float32)
        obs[0] = self.SOC
        obs[1] = (self.Tep_a-25)/(60-25)
        obs[2] = self.I_batt/50
        obs[3] = self.P_mot/150000      # [-1, 1]
        return obs
    
    def execute(self, action, car_spd, car_acc):
        # power demand
        T_axle, W_axle = self.EGS.T_W_axle(car_spd, car_acc)
        self.P_mot, T_mot, W_mot, Mot_eff = self.EGS.motor_power(T_axle, W_axle)     # unit W
        eng_power = abs(action)*1000  # unit W
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
            SoC_r = self.SOC_origin-lamda*self.follow_x  # follow_x should be in state space
            delta_SoC = self.SOC-SoC_r
            w1 = 10
            # if self.SOC > 0.90 or self.SOC < 0.10:
            #     w1 *= 5
        cost_SOC = w1*abs(delta_SoC)
        # reward about SOH
        w2 = 13000  # SoH reward coefficient    ￥13042.5 for replacing a new battery
        cost_SOH = w2*self.dsoh
        # reward about P_batt, too big is not good
        w3 = 1  # self.battery_power_penalty: [0, 1],  3-->1
        cost_power = w3*self.battery_power_penalty  # [0, w3]
        # total cost and reward
        bttery_reward = -(cost_SOC+cost_SOH)  # [-1, 0]
        reward = -fuel_cost+bttery_reward-cost_power
        self.info.update({'battery_reward': bttery_reward, 'delta_SOC': delta_SoC,
                          'cost_SOC': cost_SOC, 'cost_SOH': cost_SOH, 'cost_power': cost_power,
                          'energy_reward': reward})
        return reward
    
    def get_info(self):
        return self.info
    
    def get_done(self):
        return self.done
