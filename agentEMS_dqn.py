import numpy as np
from common.FCHEV import FCHEV_model


class EMS:
    def __init__(self):
        self.done = False
        self.info = {}
        self.FCHEV = FCHEV_model()
        self.obs_num = 1  # soc, P_mot
        self.action_num = 1  # P_FC
        self.SOC_init = 0.5
        self.SOC_target = 0.3
        self.P_mot_max = 200000  # W
        self.SOC = self.SOC_init
        self.P_mot = 0
        self.h2_fcs = 0
        self.P_batt = 0
        self.P_FCS = 0
        self.P_FCS_jerk = 0
        self.SOC_delta = 0
    
    def reset_obs(self):
        self.SOC = self.SOC_init
        self.P_mot = 0
        self.P_FCS = 0
        self.done = False
        self.info = {}
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.SOC
        # obs[1] = self.P_mot/self.P_mot_max
        return obs
    
    def execute(self, action, car_spd, car_acc):
        P_fc_action = abs(action) * self.FCHEV.P_FC_max     # kW
        # P_fc_action = abs(action)       # [0,60] kW
        self.P_FCS_jerk = abs(self.P_FCS - P_fc_action)/self.FCHEV.P_FC_max  # [0, 1]
        self.P_FCS = P_fc_action
        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(car_spd, car_acc)
        T_mot, W_mot, mot_eff, self.P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(self.P_FCS)      # kW
        self.P_batt = self.P_mot - P_dcdc*1000        # W
        self.SOC_delta, self.SOC, self.done, info_batt = self.FCHEV.run_power_battery(self.P_batt, self.SOC)
        
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': self.P_mot/1000})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        obs[0] = self.SOC
        # obs[1] = self.P_mot/self.P_mot_max
        return obs
    
    def get_reward(self):
        if self.P_batt > 0:
            h2_batt = self.P_batt / 1000 * 0.0164      # g in one second
            # 取FCS效率最高点计算, 该系数为0.164, 原为0.0167
        else:
            h2_batt = 0
        h2_equal = float(self.h2_fcs + h2_batt)
        w_h2 = 5.0
        h2_cost = float(w_h2 * h2_equal)
        w_P_FCS_jerk = 0.0
        P_FCS_cost = w_P_FCS_jerk * self.P_FCS_jerk
        w_soc = 0.0
        # if 0.3 <= self.SOC <= 0.6:
        #     soc_cost = 0
        # else:
        #     soc_cost = w_soc*min(abs(self.SOC-0.6), abs(self.SOC-0.3))
        soc_cost = w_soc * abs(self.SOC - self.SOC_target)
        # reward = -(P_FCS_cost + h2_cost + soc_cost)       # reward approach I
        
        # cost-optimal
        elec_price = 1.4        # price of electricity charge, ￥ per kWh
        if self.SOC_delta < 0:
            elec_money_spent = abs(self.SOC_delta) * self.FCHEV.C_batt * elec_price
            elec_money_revised = elec_money_spent
        else:
            elec_money_spent = 0.0        # 是否添加如下补偿？
            elec_money_revised = -abs(self.SOC_delta) * self.FCHEV.C_batt * elec_price
        h2_price = 55/1000      # ￥ per g
        h2_money = h2_price * self.h2_fcs
        total_money_spent = h2_money+elec_money_spent
        total_money_revised = h2_money+elec_money_revised
        # reward = -total_money_spent     # reward approach II
        reward = -total_money_revised       # reward approach III

        reward = float(reward)
        self.info.update({'h2_cost': h2_cost, 'P_FCS_cost': P_FCS_cost, 'EMS_reward': reward, 'h2_money': h2_money,
                          'h2_equal': h2_equal, 'h2_batt': float(h2_batt), 'soc_cost': float(soc_cost),
                          'elec_money_spent': elec_money_spent, 'elec_money_revised': elec_money_revised,
                          'total_money_spent': total_money_spent, 'total_money_revised': total_money_revised})
        return reward

    def get_info(self):
        return self.info

    def get_done(self):
        return self.done
    