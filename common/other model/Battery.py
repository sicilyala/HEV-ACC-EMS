# Designed by Jingda Wu, PhD student in MAE, NTU

import numpy as np
import math
import pickle
import scipy.io as scio
from scipy import interpolate

a = scio.loadmat('ocv_list.mat')
soc_data = a['soc_list'].squeeze()
ocv_data = a['ocv_list'].squeeze()
interp = interpolate.interp1d(soc_data, ocv_data)

"""
问题1，控制动作是电流，如何保证输出功率满足车辆需求；
2，如何应用到不同型号的电池上，直接输入对应的soc-ocv特性曲线吗，那如何设定电池容量、电压、电流设置等参数？
3，这是一个单独的智能体？
"""

class Battery:
    
    def __init__(self, init_temperature, init_soc, init_soh, target_soc=0.9,
                 max_soc=1, min_soc=0.01, max_current_scale=6,
                 min_cell_voltage=2, max_cell_voltage=6.6, trainable=True):
        
        self.timestep = 1
        self.trainable = trainable
        self.tf = init_temperature
        self.init_soc = init_soc
        self.init_soh = init_soh
        self.soh_scale = 0
        self.target_soc = target_soc
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.max_v_cell = max_cell_voltage
        self.min_v_cell = min_cell_voltage
        
        self.max_current_scale = max_current_scale
        
        self.previous_action = 0
        
        self.restart()
    
    def restart(self):
        self.count = 0  # time counting
        
        self.current = 0  # unit, A
        self.crate = 0
        self.voltage = 0  # unit, V
        self.power_out = self.current*self.voltage  # unit, W
        
        ran = 0 if self.trainable else 0  # determine whether the initial soc is random from 0.3-0.9 or determined init_soc value
        self.soc = np.random.randint(3, 9)/10 if ran else self.init_soc
        self.soh = self.init_soh
        self.soh_deriv = 0
        
        self.previous_action = 0
        
        self.v1 = 0
        self.v2 = 0
        self.ts = self.tf  # inital surface temperature. unit, Calsius
        self.tc = self.tf  # inital central temperature. unit, Calsius
        self.ta = self.tf  # inital average temperature. unit, Calsius
        
        # the following defines battery-related parameters
        self.b_list = pickle.load(open('B.pkl', 'rb'))
        
        self.r0 = 0.0031  # used in electric model
        self.r1 = 0.0062
        self.r2 = 0.0054
        self.c1 = 8710.0
        self.c2 = 258.4211
        
        self.Cn = 2.422  # used in heating model
        self.Cc = 62.7
        self.Cs = 41
        self.Ru = 5.095
        self.Rc = 1.94
        
        self.z = 0.55  # used in ageing model
        self.ideal_gas_constant = 8.31
    
    def run_step(self, action):
        self.current = action*self.max_current_scale*self.Cn
        
        self.current = 0*self.current if self.current < 0.09 else self.current  # debug 1
        
        # self.ocv = fsolve(self.f_ocv,0)
        self.ocv = self.ocv_list(self.soc)
        
        self.b = self.b_list(self.soc)      # todo:   no use
        
        cell_heat = self.current*(self.v1+self.v2+self.r0*self.current)     # H(t)
        
        total_Qbatt = self.ocv*self.Cn       # todo: what?
        
        self.crate = abs(self.current/self.Cn)      # Cn is battery capacity, Ah
        
        self.voltage = self.ocv+self.v1+self.v2+self.r0*self.current
        
        self.power_out = self.voltage*self.current
        
        soc_deriv = self.timestep*(self.current/3600/self.Cn)
        v1_deriv = self.timestep*(-self.v1/self.r1/self.c1+self.current/self.c1)
        v2_deriv = self.timestep*(-self.v2/self.r2/self.c2+self.current/self.c2)
        tc_deriv = self.timestep*(((self.ts-self.tc)/self.Rc+cell_heat)/self.Cc)
        ts_deriv = self.timestep*(((self.tc-self.ts)/self.Rc+(self.tf-self.ts)/self.Ru)/self.Cs)
        
        self.soc = self.soc+soc_deriv
        self.v1 = self.v1+v1_deriv
        self.v2 = self.v2+v2_deriv
        self.v1 = np.exp(-1/self.r1/self.c1)*self.v1+(1-np.exp(-1/self.r1/self.c1))*self.r1*self.current
        self.v2 = np.exp(-1/self.r2/self.c2)*self.v2+(1-np.exp(-1/self.r2/self.c2))*self.r2*self.current
        
        self.tc = self.tc+tc_deriv
        self.ts = self.ts+ts_deriv
        self.ta = (self.tc+self.ts)/2
        
        B = self.b_list(self.crate)
        E = 31700-370.3*self.crate
        T = self.ts+273     # TODO: Ta?
        A = (20/B/math.exp(-E/self.ideal_gas_constant/T))**(1/self.z)
        N = self.ocv*A/total_Qbatt      # TODO: 3600?
        
        self.soh_deriv = self.timestep*(abs(self.power_out)/1000/2/N/total_Qbatt)
        
        N1 = 3600*A/self.Cn
        dsoh = self.timestep*(abs(self.current/2/N1/self.Cn))
        
        self.soh = self.soh-dsoh
        
        if self.count > 0:
            self.ocv = self.ocv_list(self.soc)
            self.voltage = self.ocv+self.v1+self.v2+self.r0*self.current
            self.power_out = self.voltage*self.current
        
        next_observation = self.obtain_observation()
        
        other_variants = {'ta': self.ta, 'tc': self.tc, 'v1': self.v1, 'v2': self.v2, 'v1_deriv': v1_deriv,
                          'v2_deriv': v2_deriv,
                          'crate': self.crate, 'soh_deriv': self.soh_deriv, 'v': self.voltage,
                          'tc_deriv': tc_deriv, 'ts_deriv': ts_deriv, 'cell_heat': cell_heat, 'A': A, 'N': N1,
                          'soc_deriv': soc_deriv, 'ocv': self.ocv, 'r0': self.r0, 'soh': self.soh}
        
        reward_safety_voltage = - 0.7*(self.voltage > 3.6)-0.55*(self.voltage < 3.35)*(self.soc < 0.93)
        reward_safety_heat = 4*(45-self.tc)*(self.tc > 45)-0.03*(self.ta-10)*(self.ta < 10)
        reward_soc = -2*abs(self.target_soc-self.soc)*(self.soc <= self.target_soc)-2000*(self.soc-self.target_soc)**2*(
                    self.soc > self.target_soc-0.02)
        reward_deriv = (abs(self.previous_action-self.current) > 0.01)*(-0.02)
        reward_soh = -self.soh_deriv*self.soh_scale
        reward = 0.1+reward_safety_voltage+reward_safety_heat+reward_soc+reward_deriv+reward_soh
        
        # debug 2
        if self.soc > self.max_soc:
            self.soc = self.soc*0+1
        
        fail = self.soc > self.max_soc or self.soc < self.min_soc or self.voltage < self.min_v_cell or self.voltage > self.max_v_cell
        
        self.count += 1
        
        self.previous_action = self.current if self.count > 0 else 0
        
        finish = self.count > 2000/self.timestep
        
        done = finish or fail
        
        return next_observation, reward, done, self.count, other_variants
    
    def obtain_observation(self):
        observations = [self.soc, self._sigmoid(self.tc/50), (self.voltage-2)/4]
        return observations
    
    def _sigmoid(self, x, sigma=2):
        return 2/(1+np.exp(-sigma*x))-1
    
    def ocv_list(self, z):
        return interp(z)
    
    def f_ocv(self, ocv):
        return 0.362*np.arctan(2*(ocv-3.305)/0.007)+0.23*np.arctan(2*(ocv-3.238)/0.078)+0.196*np.arctan(
            2*(ocv-3.34)/0.003)+1.254-self.soc*2.422