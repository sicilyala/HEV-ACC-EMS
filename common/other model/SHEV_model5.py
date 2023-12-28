# -*- coding: utf-8 -*-
"""
the Model of Series hybrid energy vehicle

"""
import numpy as np
import math
import pickle
from scipy.interpolate import interp1d, interp2d
import scipy.io as scio

class SHEV_model:
    def __init__(self):
        self.I_max = 460  # Battery current limitation
        self.Q_batt = 25  # Ah
        # Nominal battery accumulate Ah-throughput
        self.nominal_life = pow(20/self.severity_factor_fun(0.35, 2.7, 41), 1/0.57)     # 48000 Ah
        self.accumulated_Ah = 0
        # P_batt_by_EGS = 0
        # self.power_data = "D:/SEU2/Program1/MADDPG-program/common/power_data.npy"
        # np.save(self.power_data, [P_batt_by_EGS])
        self._func_init()
        
        self.timestep = 1
        self.r0 = 0.0031  # used in electric model  Rs
        self.r1 = 0.0062
        self.r2 = 0.0054
        self.c1 = 8710.0
        self.c2 = 258.4211
        
        self.Cn = 2.422  # used in heating model
        self.Cc = 62.7
        self.Cs = 41
        self.Ru = 5.095
        self.Rc = 1.94
        
        # self.z = 0.55  # used in ageing model
        # self.ideal_gas_constant = 8.31
    
    def _func_init(self):
        RPM_2_rads = 2*math.pi/60
        data_dir = "D:/SEU2/Program1/MADDPG-program/common/data/"
        """
        # initial approch
        Eng_w_list = [0, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500]
        Eng_w_list = [i*36/25*RPM_2_rads for i in Eng_w_list]
        Eng_w_list = np.array(Eng_w_list)  # shape(n,)
        Eng_w_list = Eng_w_list[np.newaxis, :]  # shape(1, 10)
        Eng_t_list = [0, 10, 85, 130, 180, 220, 277, 350, 425, 500, 550, 600, 650, 664.5]
        Eng_t_list = [i*306/665 for i in Eng_t_list]
        Eng_t_list = np.array(Eng_t_list)
        Eng_t_list = Eng_t_list[np.newaxis, :]  # shape(1, 14)
        data_path = data_dir+'engine_eff1.mat'
        data = scio.loadmat(data_path)
        Eng_bsfc_map = data['be']
        Eng_fuel_map = Eng_bsfc_map*(Eng_w_list*Eng_t_list.T)/3600
        self.Eng_fuel_func = interp2d(Eng_w_list, Eng_t_list, Eng_fuel_map)
        """
        # Engine efficiency, my opinion
        data_path = data_dir+'be.mat'
        data = scio.loadmat(data_path)
        be = data['be']
        data_path = data_dir+'W_e.mat'
        data = scio.loadmat(data_path)
        We = data['W_e']
        data_path = data_dir+'T_e.mat'
        data = scio.loadmat(data_path)
        Te = data['T_e']
        self.Eng_fuel_eff_map = interp2d(We[0, :], Te[:, 0], be)
        
        Eng_pwr_opt_list = [16351, 13277, 12960, 12354, 11928, 4366, 13064, 13987, 47848, 34993, 29996, 21998, 24000,
                            27998, 48005, 47844, 48052, 52907, 53781, 55146, 56816, 62000]
        w_list = [83, 71, 71, 71, 72, 71, 71, 74, 172, 141, 127, 111, 117, 122, 176, 172, 174, 191, 197, 202, 212, 226]
        self.optimal_spd = interp1d(Eng_pwr_opt_list, w_list, kind='linear', fill_value='extrapolate')
        Eng_t_maxlist = [130, 240, 260, 300, 302, 300, 280, 265, 248]
        Eng_w_maxlist = [0, 130, 150, 200, 250, 300, 325, 350, 377]
        self.Eng_maxtrq_func = interp1d(Eng_w_maxlist, Eng_t_maxlist)
        
        # motor
        Mot_w_list = [0, 27.6381909547739, 55.2763819095477, 82.9145728643216, 110.552763819095, 138.190954773869,
                      165.829145728643, 193.467336683417, 221.105527638191, 248.743718592965, 276.381909547739,
                      304.020100502513, 331.658291457286, 359.296482412060, 386.934673366834, 414.572864321608,
                      442.211055276382, 469.849246231156, 497.487437185930, 525.125628140704, 552.763819095477,
                      580.402010050251, 608.040201005025, 635.678391959799, 663.316582914573, 690.954773869347,
                      718.592964824121, 746.231155778895, 773.869346733668, 801.507537688442, 829.145728643216,
                      856.783919597990, 884.422110552764, 912.060301507538, 939.698492462312, 967.336683417085,
                      994.974874371859, 1022.61306532663, 1050.25125628141, 1077.88944723618, 1105.52763819095,
                      1133.16582914573, 1160.80402010050, 1188.44221105528, 1216.08040201005, 1243.71859296482,
                      1271.35678391960, 1298.99497487437, 1326.63316582915, 1354.27135678392, 1381.90954773869,
                      1409.54773869347, 1437.18592964824, 1464.82412060302, 1492.46231155779, 1520.10050251256,
                      1547.73869346734, 1575.37688442211, 1603.01507537688, 1630.65326633166, 1658.29145728643,
                      1685.92964824121, 1713.56783919598, 1741.20603015075, 1768.84422110553, 1796.48241206030,
                      1824.12060301508, 1851.75879396985, 1879.39698492462, 1907.03517587940, 1934.67336683417,
                      1962.31155778894, 1989.94974874372, 2017.58793969849, 2045.22613065327, 2072.86432160804,
                      2100.50251256281, 2128.14070351759, 2155.77889447236, 2183.41708542714, 2211.05527638191,
                      2238.69346733668, 2266.33165829146, 2293.96984924623, 2321.60804020101, 2349.24623115578,
                      2376.88442211055, 2404.52261306533, 2432.16080402010, 2459.79899497487, 2487.43718592965,
                      2515.07537688442, 2542.71356783920, 2570.35175879397, 2597.98994974874, 2625.62814070352,
                      2653.26633165829, 2680.90452261307, 2708.54271356784, 2736.18090452261, 2763.81909547739,
                      2791.45728643216, 2819.09547738694, 2846.73366834171, 2874.37185929648, 2902.01005025126,
                      2929.64824120603, 2957.28643216080, 2984.92462311558, 3012.56281407035, 3040.20100502513,
                      3067.83919597990, 3095.47738693467, 3123.11557788945, 3150.75376884422, 3178.39195979900,
                      3206.03015075377, 3233.66834170854, 3261.30653266332, 3288.94472361809, 3316.58291457286,
                      3344.22110552764, 3371.85929648241, 3399.49748743719, 3427.13567839196, 3454.77386934673,
                      3482.41206030151, 3510.05025125628, 3537.68844221106, 3565.32663316583, 3592.96482412060,
                      3620.60301507538, 3648.24120603015, 3675.87939698493, 3703.51758793970, 3731.15577889447,
                      3758.79396984925, 3786.43216080402, 3814.07035175879, 3841.70854271357, 3869.34673366834,
                      3896.98492462312, 3924.62311557789, 3952.26130653266, 3979.89949748744, 4007.53768844221,
                      4035.17587939699, 4062.81407035176, 4090.45226130653, 4118.09045226131, 4145.72864321608,
                      4173.36683417085, 4201.00502512563, 4228.64321608040, 4256.28140703518, 4283.91959798995,
                      4311.55778894472, 4339.19597989950, 4366.83417085427, 4394.47236180905, 4422.11055276382,
                      4449.74874371859, 4477.38693467337, 4505.02512562814, 4532.66331658291, 4560.30150753769,
                      4587.93969849246, 4615.57788944724, 4643.21608040201, 4670.85427135678, 4698.49246231156,
                      4726.13065326633, 4753.76884422111, 4781.40703517588, 4809.04522613065, 4836.68341708543,
                      4864.32160804020, 4891.95979899498, 4919.59798994975, 4947.23618090452, 4974.87437185930,
                      5002.51256281407, 5030.15075376884, 5057.78894472362, 5085.42713567839, 5113.06532663317,
                      5140.70351758794, 5168.34170854271, 5195.97989949749, 5223.61809045226, 5251.25628140704,
                      5278.89447236181, 5306.53266331658, 5334.17085427136, 5361.80904522613, 5389.44723618091,
                      5417.08542713568, 5444.72361809045, 5472.36180904523, 5500]
        Mot_w_list = [i*72/60*RPM_2_rads for i in Mot_w_list]
        Mot_w_list = np.array(Mot_w_list)
        Mot_t_list = [-853.510000000000, -844.919899497487, -836.329798994975, -827.739698492462, -819.149597989950,
                      -810.559497487437, -801.969396984925, -793.379296482412, -784.789195979899, -776.199095477387,
                      -767.608994974874, -759.018894472362, -750.428793969849, -741.838693467337, -733.248592964824,
                      -724.658492462312, -716.068391959799, -707.478291457286, -698.888190954774, -690.298090452261,
                      -681.707989949749, -673.117889447236, -664.527788944724, -655.937688442211, -647.347587939699,
                      -638.757487437186, -630.167386934674, -621.577286432161, -612.987185929648, -604.397085427136,
                      -595.806984924623, -587.216884422111, -578.626783919598, -570.036683417085, -561.446582914573,
                      -552.856482412060, -544.266381909548, -535.676281407035, -527.086180904523, -518.496080402010,
                      -509.905979899498, -501.315879396985, -492.725778894472, -484.135678391960, -475.545577889447,
                      -466.955477386935, -458.365376884422, -449.775276381910, -441.185175879397, -432.595075376884,
                      -424.004974874372, -415.414874371859, -406.824773869347, -398.234673366834, -389.644572864322,
                      -381.054472361809, -372.464371859297, -363.874271356784, -355.284170854271, -346.694070351759,
                      -338.103969849246, -329.513869346734, -320.923768844221, -312.333668341709, -303.743567839196,
                      -295.153467336684, -286.563366834171, -277.973266331658, -269.383165829146, -260.793065326633,
                      -252.202964824121, -243.612864321608, -235.022763819096, -226.432663316583, -217.842562814070,
                      -209.252462311558, -200.662361809045, -192.072261306533, -183.482160804020, -174.892060301508,
                      -166.301959798995, -157.711859296482, -149.121758793970, -140.531658291457, -131.941557788945,
                      -123.351457286432, -114.761356783920, -106.171256281407, -97.5811557788945, -88.9910552763820,
                      -80.4009547738694, -71.8108542713569, -63.2207537688443, -54.6306532663317, -46.0405527638192,
                      -37.4504522613066, -28.8603517587941, -20.2702512562814, -11.6801507537689, -3.09005025125634,
                      5.50005025125620, 14.0901507537689, 22.6802512562814, 31.2703517587938, 39.8604522613064,
                      48.4505527638190, 57.0406532663316, 65.6307537688441, 74.2208542713566, 82.8109547738693,
                      91.4010552763818, 99.9911557788944, 108.581256281407, 117.171356783919, 125.761457286432,
                      134.351557788945, 142.941658291457, 151.531758793970, 160.121859296482, 168.711959798995,
                      177.302060301507, 185.892160804020, 194.482261306533, 203.072361809045, 211.662462311558,
                      220.252562814070, 228.842663316583, 237.432763819096, 246.022864321608, 254.612964824121,
                      263.203065326633, 271.793165829146, 280.383266331658, 288.973366834171, 297.563467336683,
                      306.153567839196, 314.743668341708, 323.333768844221, 331.923869346733, 340.513969849246,
                      349.104070351759, 357.694170854271, 366.284271356784, 374.874371859296, 383.464472361809,
                      392.054572864321, 400.644673366834, 409.234773869347, 417.824874371859, 426.414974874372,
                      435.005075376884, 443.595175879397, 452.185276381910, 460.775376884422, 469.365477386935,
                      477.955577889447, 486.545678391960, 495.135778894472, 503.725879396985, 512.315979899498,
                      520.906080402010, 529.496180904523, 538.086281407035, 546.676381909548, 555.266482412060,
                      563.856582914573, 572.446683417085, 581.036783919598, 589.626884422111, 598.216984924623,
                      606.807085427136, 615.397185929648, 623.987286432161, 632.577386934673, 641.167487437186,
                      649.757587939698, 658.347688442211, 666.937788944724, 675.527889447236, 684.117989949749,
                      692.708090452261, 701.298190954774, 709.888291457286, 718.478391959799, 727.068492462312,
                      735.658592964824, 744.248693467337, 752.838793969849, 761.428894472362, 770.018994974874,
                      778.609095477387, 787.199195979899, 795.789296482412, 804.379396984925, 812.969497487437,
                      821.559597989950, 830.149698492462, 838.739798994975, 847.329899497487,
                      855.920000000000]
        Mot_t_list = [i*320/830 for i in Mot_t_list]
        Mot_t_list = np.array(Mot_t_list)
        Mot_w_maxlist = [0, 265.29, 691.15]
        Mot_t_maxlist = [326.25, 326.62, 161.276]
        Mot_t_minlist = [0, -55, - 160, - 235, - 270, - 305, - 323.44, - 324.4, - 325.3, - 325.5, - 325.6, - 319,
                         - 316.28, - 239.37, - 160.5]
        Mot_w_minlist = [0, 10.85, 31.8, 45.8, 52.6, 59.6, 61, 62, 85, 130, 223, 237, 266, 479, 691]
        data_path = data_dir+'motor_eff1.mat'
        data = scio.loadmat(data_path)
        Mot_eta_map = data['bm']/100
        Mot_eta_map = np.array(Mot_eta_map)
        # Mot_eta_quarter = data['bm'] / 100
        # Mot_eta_quarter = np.array(Mot_eta_quarter)
        # print(Mot_eta_quarter.shape)
        # Mot_eta_alltrqs = np.concatenate(([np.fliplr(Mot_eta_quarter[:, 1:]), Mot_eta_quarter]), axis=1)
        # Mot_eta_map = np.concatenate(([np.flipud(Mot_eta_alltrqs[1:, :]), Mot_eta_alltrqs]))
        self.Mot_eta_map_func = interp2d(Mot_w_list, Mot_t_list, Mot_eta_map)
        self.Mot_mintrq_func = interp1d(Mot_w_minlist, Mot_t_minlist, kind='linear', fill_value='extrapolate')
        self.Mot_maxtrq_func = interp1d(Mot_w_maxlist, Mot_t_maxlist, kind='linear', fill_value='extrapolate')
        # ISG
        Gen_w_list = [0, 25.1256281407035, 50.2512562814070, 75.3768844221106, 100.502512562814, 125.628140703518,
                      150.753768844221, 175.879396984925, 201.005025125628, 226.130653266332, 251.256281407035,
                      276.381909547739, 301.507537688442, 326.633165829146, 351.758793969849, 376.884422110553,
                      402.010050251256, 427.135678391960, 452.261306532663, 477.386934673367, 502.512562814070,
                      527.638190954774, 552.763819095477, 577.889447236181, 603.015075376884, 628.140703517588,
                      653.266331658292, 678.391959798995, 703.517587939699, 728.643216080402, 753.768844221106,
                      778.894472361809, 804.020100502513, 829.145728643216, 854.271356783920, 879.396984924623,
                      904.522613065327, 929.648241206030, 954.773869346734, 979.899497487437, 1005.02512562814,
                      1030.15075376884, 1055.27638190955, 1080.40201005025, 1105.52763819095, 1130.65326633166,
                      1155.77889447236, 1180.90452261307, 1206.03015075377, 1231.15577889447, 1256.28140703518,
                      1281.40703517588, 1306.53266331658, 1331.65829145729, 1356.78391959799, 1381.90954773869,
                      1407.03517587940, 1432.16080402010, 1457.28643216080, 1482.41206030151, 1507.53768844221,
                      1532.66331658291, 1557.78894472362, 1582.91457286432, 1608.04020100503, 1633.16582914573,
                      1658.29145728643, 1683.41708542714, 1708.54271356784, 1733.66834170854, 1758.79396984925,
                      1783.91959798995, 1809.04522613065, 1834.17085427136, 1859.29648241206, 1884.42211055276,
                      1909.54773869347, 1934.67336683417, 1959.79899497487, 1984.92462311558, 2010.05025125628,
                      2035.17587939699, 2060.30150753769, 2085.42713567839, 2110.55276381910, 2135.67839195980,
                      2160.80402010050, 2185.92964824121, 2211.05527638191, 2236.18090452261, 2261.30653266332,
                      2286.43216080402, 2311.55778894472, 2336.68341708543, 2361.80904522613, 2386.93467336683,
                      2412.06030150754, 2437.18592964824, 2462.31155778895, 2487.43718592965, 2512.56281407035,
                      2537.68844221106, 2562.81407035176, 2587.93969849246, 2613.06532663317, 2638.19095477387,
                      2663.31658291457, 2688.44221105528, 2713.56783919598, 2738.69346733668, 2763.81909547739,
                      2788.94472361809, 2814.07035175879, 2839.19597989950, 2864.32160804020, 2889.44723618090,
                      2914.57286432161, 2939.69849246231, 2964.82412060302, 2989.94974874372, 3015.07537688442,
                      3040.20100502513, 3065.32663316583, 3090.45226130653, 3115.57788944724, 3140.70351758794,
                      3165.82914572864, 3190.95477386935, 3216.08040201005, 3241.20603015075, 3266.33165829146,
                      3291.45728643216, 3316.58291457286, 3341.70854271357, 3366.83417085427, 3391.95979899498,
                      3417.08542713568, 3442.21105527638, 3467.33668341709, 3492.46231155779, 3517.58793969849,
                      3542.71356783920, 3567.83919597990, 3592.96482412060, 3618.09045226131, 3643.21608040201,
                      3668.34170854271, 3693.46733668342, 3718.59296482412, 3743.71859296482, 3768.84422110553,
                      3793.96984924623, 3819.09547738694, 3844.22110552764, 3869.34673366834, 3894.47236180905,
                      3919.59798994975, 3944.72361809045, 3969.84924623116, 3994.97487437186, 4020.10050251256,
                      4045.22613065327, 4070.35175879397, 4095.47738693467, 4120.60301507538, 4145.72864321608,
                      4170.85427135678, 4195.97989949749, 4221.10552763819, 4246.23115577889, 4271.35678391960,
                      4296.48241206030, 4321.60804020101, 4346.73366834171, 4371.85929648241, 4396.98492462312,
                      4422.11055276382, 4447.23618090452, 4472.36180904523, 4497.48743718593, 4522.61306532663,
                      4547.73869346734, 4572.86432160804, 4597.98994974874, 4623.11557788945, 4648.24120603015,
                      4673.36683417085, 4698.49246231156, 4723.61809045226, 4748.74371859297, 4773.86934673367,
                      4798.99497487437, 4824.12060301508, 4849.24623115578, 4874.37185929648, 4899.49748743719,
                      4924.62311557789, 4949.74874371859, 4974.87437185930, 5000]
        Gen_w_list = [i*4/5*RPM_2_rads for i in Gen_w_list]
        Gen_t_list = [-482.980000000000, -478.169949748744, -473.359899497487, -468.549849246231, -463.739798994975,
                      -458.929748743719, -454.119698492462, -449.309648241206, -444.499597989950, -439.689547738694,
                      -434.879497487437, -430.069447236181, -425.259396984925, -420.449346733668, -415.639296482412,
                      -410.829246231156, -406.019195979900, -401.209145728643, -396.399095477387, -391.589045226131,
                      -386.778994974874, -381.968944723618, -377.158894472362, -372.348844221106, -367.538793969849,
                      -362.728743718593, -357.918693467337, -353.108643216080, -348.298592964824, -343.488542713568,
                      -338.678492462312, -333.868442211055, -329.058391959799, -324.248341708543, -319.438291457286,
                      -314.628241206030, -309.818190954774, -305.008140703518, -300.198090452261, -295.388040201005,
                      -290.577989949749, -285.767939698492, -280.957889447236, -276.147839195980, -271.337788944724,
                      -266.527738693467, -261.717688442211, -256.907638190955, -252.097587939698, -247.287537688442,
                      -242.477487437186, -237.667437185930, -232.857386934673, -228.047336683417, -223.237286432161,
                      -218.427236180905, -213.617185929648, -208.807135678392, -203.997085427136, -199.187035175879,
                      -194.376984924623, -189.566934673367, -184.756884422111, -179.946834170854, -175.136783919598,
                      -170.326733668342, -165.516683417085, -160.706633165829, -155.896582914573, -151.086532663317,
                      -146.276482412060, -141.466432160804, -136.656381909548, -131.846331658291, -127.036281407035,
                      -122.226231155779, -117.416180904523, -112.606130653266, -107.796080402010, -102.986030150754,
                      -98.1759798994975, -93.3659296482413, -88.5558793969849, -83.7458291457286, -78.9357788944724,
                      -74.1257286432161, -69.3156783919598, -64.5056281407035, -59.6955778894472, -54.8855276381910,
                      -50.0754773869347, -45.2654271356784, -40.4553768844221, -35.6453266331658, -30.8352763819095,
                      -26.0252261306533, -21.2151758793970, -16.4051256281407, -11.5950753768844, -6.78502512562812,
                      -1.97497487437187, 2.83507537688445, 7.64512562814070, 12.4551758793970, 17.2652261306533,
                      22.0752763819095, 26.8853266331658, 31.6953768844221, 36.5054271356784, 41.3154773869346,
                      46.1255276381910, 50.9355778894472, 55.7456281407035, 60.5556783919598, 65.3657286432161,
                      70.1757788944724, 74.9858291457286, 79.7958793969849, 84.6059296482413, 89.4159798994975,
                      94.2260301507538, 99.0360804020100, 103.846130653266, 108.656180904523, 113.466231155779,
                      118.276281407035, 123.086331658291, 127.896381909548, 132.706432160804, 137.516482412060,
                      142.326532663317, 147.136582914573, 151.946633165829, 156.756683417085, 161.566733668342,
                      166.376783919598, 171.186834170854, 175.996884422111, 180.806934673367, 185.616984924623,
                      190.427035175879, 195.237085427136, 200.047135678392, 204.857185929648, 209.667236180905,
                      214.477286432161, 219.287336683417, 224.097386934673, 228.907437185930, 233.717487437186,
                      238.527537688442, 243.337587939699, 248.147638190955, 252.957688442211, 257.767738693467,
                      262.577788944724, 267.387839195980, 272.197889447236, 277.007939698492, 281.817989949749,
                      286.628040201005, 291.438090452261, 296.248140703518, 301.058190954774, 305.868241206030,
                      310.678291457286, 315.488341708543, 320.298391959799, 325.108442211055, 329.918492462312,
                      334.728542713568, 339.538592964824, 344.348643216080, 349.158693467337, 353.968743718593,
                      358.778793969849, 363.588844221106, 368.398894472362, 373.208944723618, 378.018994974874,
                      382.829045226131, 387.639095477387, 392.449145728643, 397.259195979899, 402.069246231156,
                      406.879296482412, 411.689346733668, 416.499396984925, 421.309447236181, 426.119497487437,
                      430.929547738694, 435.739597989950, 440.549648241206, 445.359698492462, 450.169748743719,
                      454.979798994975, 459.789849246231, 464.599899497488, 469.409949748744, 474.220000000000]
        Gen_t_list = [i*370/450 for i in Gen_t_list]
        Gen_w_minlist = [0, 100, 200, 210, 214, 216, 218, 220, 250, 300, 350, 400, 419]
        Gen_t_minlist = [-396, - 396, - 396, - 396, - 396, - 394, - 392, - 388, - 352, - 289, - 230, - 171, - 145]
        self.Gen_mintrq_func = interp1d(Gen_w_minlist, Gen_t_minlist, kind='linear', fill_value='extrapolate')
        data_path = data_dir+'isg_eff.mat'
        data = scio.loadmat(data_path)
        Gen_eta_map = data['bg']/100
        Gen_eta_map = np.array(Gen_eta_map)
        # Gen_eta_quarter = data['bg'] / 100
        # Gen_eta_alltrqs = np.concatenate((Gen_eta_quarter[:, 1:], Gen_eta_quarter), axis=1)
        # Gen_eta_map = np.concatenate(([np.flipud(Gen_eta_alltrqs[1:, :]), Gen_eta_alltrqs]))
        self.Gen_eta_func = interp2d(Gen_w_list, Gen_t_list, Gen_eta_map)
        
        # Battery
        # SoC
        SOC_list = [0, 0.00132660038355303, 0.0574708841412090, 0.107025590196296, 0.156580329178506, 0.206134478930137,
                    0.255687515404994, 0.305240257178054, 0.354795683551491, 0.404476817165514, 0.454031523456938,
                    0.503585181935343, 0.553137334622058, 0.602688668725870, 0.652241410558679, 0.701796476729142,
                    0.751350724430672, 0.800904841303918, 0.850591981005177, 0.900297369762374, 0.950130539134393,
                    0.999900000000000, 1]
        # Discharging resistance
        R_dis = [0.313944125557612, 0.313944125557612, 0.313944125557612, 0.242065812972601, 0.230789057570884,
                 0.212882084646638, 0.196703117398054, 0.191970726117600, 0.182268278911152, 0.176602004689331,
                 0.173003239603114, 0.172594564949654, 0.177777197273542, 0.188139855710787, 0.212047123247474,
                 0.218402021824003, 0.211830447697718, 0.203822955457752, 0.197028250778952, 0.194459010902986,
                 0.218377885152249, 0.296468469580613, 0.296468469580613]  # ohm
        # Charging resistance
        R_chg = [0.278440842615743, 0.278440842615743, 0.278440842615743, 0.220685921062183, 0.219189603825182,
                 0.205347807991030, 0.191299532644795, 0.187277234013650, 0.178094888469773, 0.172984724665031,
                 0.169768580157996, 0.169487135972806, 0.174936544546881, 0.185495927766875, 0.208445003365321,
                 0.213703074103948, 0.206748591485580, 0.198442444892965, 0.191938639000188, 0.189976229191598,
                 0.214377935413427, 0.296464707815351, 0.296464707815351]
        # Open circuit voltage
        V_oc = [491.8387067, 491.8387067, 517.6726653, 521.6434164, 527.0477514, 531.5501691, 535.3424403,
                539.2562079, 541.5306725, 543.354122, 545.2170669, 547.3269911, 549.9472871, 553.641686,
                559.2529919, 564.8908875, 570.3695838, 576.2404755, 582.5029103, 589.1188686, 596.2324457,
                604.3783404, 604.3783404]
        V_oc = [(i+15)*360/570 for i in V_oc]  # TODO: why?
        self.R_dis_func = interp1d(SOC_list, R_dis, kind='linear', fill_value='extrapolate')
        self.R_chg_func = interp1d(SOC_list, R_chg, kind='linear', fill_value='extrapolate')
        self.V_func = interp1d(SOC_list, V_oc, kind='linear', fill_value='extrapolate')
        # from Wu Jingda
        Ic_rate = [0.5, 2, 6, 10]
        Bc_data = [31630, 21681, 12934, 15512]
        self.Bc_func = interp1d(Ic_rate, Bc_data, kind='linear', fill_value='extrapolate')
        self.ocv_func = pickle.load(open(data_dir+'ocv.pkl', 'rb'))
    
    @staticmethod
    def T_W_axle(car_spd, car_acc):
        """计算传动轴转速和扭矩"""
        # parameters of car
        wheel_radius = 0.447  # m
        mass = 3500  # 3500
        C_roll = 0.01
        density_air = 1.226  # N*s2/m4
        area_frontal = 3.9  # 3.9
        G = 9.81
        C_d = 0.65  # 0.65
        G_f = 5.857  # Final reducer ratio
        # calculate
        W_axle = car_spd/wheel_radius*G_f  # 传动轴转速
        T_axle = wheel_radius/G_f*(mass*car_acc+mass*G*C_roll+0.5*density_air*area_frontal*C_d*(car_spd**2))
        # if car_spd > 0:  # m/s      # 传动轴力矩 Nm
        #     T_axle = wheel_radius/G_f*(mass*car_acc + mass*G*C_roll + 0.5*density_air*area_frontal*C_d*(car_spd**2))
        # else:
        #     T_axle = wheel_radius/G_f*(mass*car_acc + 0.5*density_air*area_frontal*C_d*(car_spd**2))  # Nm
        return T_axle, W_axle
    
    def motor_power(self, T_axle, W_axle):
        # Motor
        T_mot = T_axle
        W_mot = W_axle
        # Motor s.t.
        if T_mot < 0:
            T_mot_edge = self.Mot_mintrq_func(abs(W_mot))
            T_mot_edge = T_mot_edge.tolist()
        else:
            T_mot_edge = self.Mot_maxtrq_func(abs(W_mot))
            T_mot_edge = T_mot_edge.tolist()
        T_mot = (T_mot < 0)*((T_mot <= T_mot_edge)*T_mot_edge+(T_mot > T_mot_edge)*T_mot) + \
                (T_mot >= 0)*((T_mot <= T_mot_edge)*T_mot+(T_mot > T_mot_edge)*T_mot_edge)
        # efficiency of the electric motor
        Mot_eff0 = (W_mot == 0)+(W_mot != 0)*self.Mot_eta_map_func(W_mot, T_mot)
        Mot_eff0 = Mot_eff0[0]
        if 0 < Mot_eff0 < 0.7633:  # minimal efficiency value of motor
            Mot_eff = 0.7633
        else:
            Mot_eff = Mot_eff0
        # Calculate power consumption
        if Mot_eff == 0:
            P_mot = 0
        else:
            p_mot = T_mot*W_mot
            if p_mot <= 0:
                P_mot = p_mot*Mot_eff
            else:
                P_mot = p_mot/Mot_eff  # motor power requested by driver
        return P_mot, T_mot, W_mot, Mot_eff
    
    def T_W_engine(self, eng_power, T_axle):
        if (eng_power < 500) or (T_axle < 0):
            # eng_power = 0
            T_eng = 0
            W_eng = 0
            return T_eng, W_eng
        # Engine
        W_eng = self.optimal_spd(eng_power)  # engine works at the opitimal operation?
        W_eng = W_eng.tolist()  # float
        T_eng = eng_power/W_eng
        
        T_eng_max = self.Eng_maxtrq_func(W_eng)  # TODO:  self.Eng_maxtrq_func
        T_eng = (T_eng < T_eng_max)*T_eng+(T_eng >= T_eng_max)*T_eng_max
        
        # if (eng_power < 500) or T_axle < 0:
        #     T_eng = 0
        #     W_eng = 0
        return T_eng, W_eng
    
    def ISG_power(self, T_eng, W_eng):
        T_ISG = T_eng
        W_ISG = W_eng
        # ISG s.t.
        T_ISG_edge = self.Gen_mintrq_func(abs(W_ISG))
        T_ISG_edge = T_ISG_edge.tolist()
        T_ISG = (T_ISG <= T_ISG_edge)*T_ISG_edge+(T_ISG > T_ISG_edge)*T_ISG
        T_eng = (T_ISG <= T_ISG_edge)*T_ISG_edge+(T_ISG > T_ISG_edge)*T_eng
        # efficiency of the electric generator
        Gen_eff = (W_ISG == 0)+(W_ISG != 0)*self.Gen_eta_func(W_ISG, T_ISG)  # list
        Gen_eff = Gen_eff[0]
        # power calculare
        P_ISG = W_ISG*T_ISG*Gen_eff  # output power provided by engine-generator set
        return P_ISG, T_ISG, W_ISG, Gen_eff, T_eng
    
    def run_EGS(self, P_eng, T_eng, W_eng):
        # power
        # T_eng, W_eng = self.T_W_engine(eng_power, T_axle)
        if P_eng < 500:
            T_eng = 0
            W_eng = 0
        T_eng_max = self.Eng_maxtrq_func(W_eng)
        T_eng = (T_eng < T_eng_max)*T_eng+(T_eng >= T_eng_max)*T_eng_max
        P_ISG, T_ISG, W_ISG, Gen_eff, T_eng = self.ISG_power(T_eng, W_eng)
        P_eng = T_eng*W_eng
        # fuel efficiency
        Eng_fuel_eff0 = self.Eng_fuel_eff_map(W_eng, T_eng)  # list
        Eng_fuel_eff = Eng_fuel_eff0[0]
        if Eng_fuel_eff < 0.2:
            Eng_fuel_eff = 0.2
        m_dot_fuel = P_eng/Eng_fuel_eff/42500000  # kg/s
        fuel_cost = m_dot_fuel/0.725  # L/s     0.725 kg/L
        
        out_info = {'engine_speed': W_eng, 'engine_torque': T_eng, 'engine_power': P_eng/1000,
                    'ISG_speed': W_ISG, 'ISG_torque': T_ISG, 'ISG_power': P_ISG/1000, 'Gen_eff': Gen_eff,
                    'Engine_fuel_efficiency': Eng_fuel_eff}
        return P_eng, T_eng, Gen_eff, Eng_fuel_eff, fuel_cost, P_ISG, out_info
    
    @staticmethod
    def severity_factor_fun(soc, I_c, T_batt):
        if soc > 0.45:
            # B = 2694.5*soc+6022.2
            B = 1385.5*soc+4193.2
        else:
            # B = 2896.6*soc+7411.2
            B = 1287.6*soc+6356.3
        E_a = 31700  # activation energy, J/mol
        R_g = 8.31  # Universal gas constant
        T_kelvin = 273.15+T_batt
        sigma = B*math.exp((163.3*I_c-E_a)/(R_g*T_kelvin))
        return sigma
    
    def run_battery(self, P_batt, SOC, Tep_c, Tep_s):
        fail = False
        # Battery efficiency
        e_batt = (P_batt > 0)+(P_batt <= 0)*0.98
        # Battery internal resistance, on package level
        resistance = (P_batt > 0)*self.R_dis_func(SOC)+(P_batt <= 0)*self.R_chg_func(SOC)
        # Battery voltage
        V_batt = self.V_func(SOC)  # OCV
        # Battery current
        if V_batt**2 < 4*resistance*P_batt:
            I_batt = e_batt*V_batt/(2*resistance)
        else:
            I_batt = e_batt*(V_batt-math.sqrt(V_batt**2-4*resistance*P_batt))/(2*resistance)
        if I_batt > self.I_max:
            I_batt = self.I_max
        I_c = abs(I_batt/self.Q_batt)  # c-rate
        SOC_new = -I_batt/self.Q_batt/3600+SOC
        if SOC_new > 1:
            SOC_new = 1.0
            fail = True
        if SOC_new < 0:
            SOC_new = 0.0
            fail = True
        self.accumulated_Ah += abs(I_batt)*self.timestep/3600  # 1s , now is on package level, should on cell
        # Battery temperature, on cell or package?
        Tep_e = 25  # 环境的温度,摄氏度
        C_c = 62.7  # heat capacity of the core,J/K
        C_s = 4.5  # heat capacity of the casing,J/K
        R_c = 1.94  # heat conduction resistance,K/W
        # R_u = 15  # convection resistance,K/W       3.19?
        R_u = 3.19
        H_gen = (I_batt**2)*resistance
        delta_Tep_c = (Tep_s-Tep_c)/(R_c*C_c)+(H_gen/C_c)
        delta_Tep_s = ((Tep_e-Tep_s)/(R_u*C_s))-((Tep_s-Tep_c)/(R_c*C_c))
        Tep_c = Tep_c+delta_Tep_c
        Tep_s = Tep_s+delta_Tep_s
        if Tep_c > 55:
            Tep_c = 55
        if Tep_s > 55:
            Tep_s = 55
        Tep_a = (Tep_c+Tep_s)/2
        '''
        # battery aging 容量损失的经验模型 : on cell level
        # paper: Optimal energy management of hybrid electric vehicles including battery aging
        # severity_factor = self._severity_factor_fun(SOC_new, abs(I_c), Tep_a)
        # Ah_eff = severity_factor*abs(I_batt)
        # life_depleted_fraction = Ah_eff/self.nominal_life
        # estimated_life = pow(20/severity_factor, 1/0.57)
        '''
        # battery aging
        # paper: A control-oriented cycle-life model for hybrid electric vehicle lithium-ion batteries
        # paper: Energy Management Strategy for HEVs Including Battery Life Optimization
        sigma = self.severity_factor_fun(SOC_new, I_c, Tep_a)
        estimated_life = pow(20/sigma, 1/0.57)
        severity_factor = self.nominal_life/estimated_life
        Ah_eff = severity_factor * abs(I_batt) * self.timestep / 3600
        life_depleted_fraction = Ah_eff/self.nominal_life       # 损失100%才报废
        
        # 吴论文的SOH模型
        # Bc = self.Bc_func(I_c)
        # E = 31700-370.3*I_c
        # T = Tep_a+273
        # Ah = (20/Bc/math.exp(-E/8.31/T))**(1/0.55)
        # N1 = 3600*Ah/self.Q_batt
        # dsoh = abs(I_batt/2/N1/self.Q_batt)  # delta_time = 1
        # SOH_new = SOH-dsoh
        # # punish dsoh
        # life_depleted_fraction = dsoh
        
        out_info = {'SoC': SOC_new, 'battery_temperature': Tep_a, 'accumulated_Ah': self.accumulated_Ah,
                    'resistance': resistance, 'battery_OCV': V_batt, 'current': I_batt, 'current_rate': I_c,
                    'severity_factor': severity_factor, 'estimated_life': estimated_life,
                    'battery_power': P_batt/1000, 'life_depleted_fraction': life_depleted_fraction,
                    'Ah_eff': Ah_eff, 'sigma': sigma}
        done = fail
        return SOC_new, Tep_c, Tep_s, Tep_a, life_depleted_fraction, I_batt, done, out_info
    
    def run_battery_cell(self, P_batt, paras_list):
        fail = False
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a, Voc, V1, V2]
        SOC = paras_list[0]
        SOH = paras_list[1]
        Tep_c = paras_list[2]
        Tep_s = paras_list[3]
        Tep_a = paras_list[4]
        Voc = paras_list[5]
        V1 = paras_list[6]
        V2 = paras_list[7]
        # battery pack of 168*6 cells
        P_cell = P_batt/(168*6)     # in W
        # print('cell power: %.4f'%P_cell)
        V_3 = Voc+V1+V2
        delta = V_3**2+4*self.r0*P_cell
        if delta < 0:
            I_batt = -V_3/(2*self.r0)
        else:
            I_batt = (-V_3+math.sqrt(delta))/(2*self.r0)
        # I_batt = (I_batt < -6)*(-6)+(-6 < I_batt < 6)*I_batt+(I_batt > 6)*6   # I_batt_range: [-6, 6] ?
        Ic_rate = abs(I_batt/self.Cn)
        cell_heat = I_batt*(V1+V2+self.r0*I_batt)  # H(t)
        soc_deriv = self.timestep*(I_batt/3600/self.Cn)
        v1_deriv = self.timestep*(-V1/self.r1/self.c1+I_batt/self.c1)
        v2_deriv = self.timestep*(-V2/self.r2/self.c2+I_batt/self.c2)
        tc_deriv = self.timestep*(((Tep_s-Tep_c)/self.Rc+cell_heat)/self.Cc)
        ts_deriv = self.timestep*(((Tep_c-Tep_s)/self.Rc+(Tep_a-Tep_s)/self.Ru)/self.Cs)
        # electric model
        SOC_new = SOC-soc_deriv
        if SOC_new > 1:
            SOC_new = 1.0
            fail = True
        if SOC_new < 0.001:
            fail = True
            SOC_new = 0.001
        V1_new = V1+v1_deriv
        V2_new = V2+v2_deriv
        # print('SOC: %.6f'%SOC_new)
        Voc_new = self.ocv_func(SOC_new)  # for a cell
        Voc_new = Voc_new*13.87/168
        # terminal voltage
        Vt_new = Voc_new+V1_new+V2_new+self.r0*I_batt
        power_out = Vt_new * I_batt
        # thermal model
        Tep_c_new = Tep_c+tc_deriv
        Tep_s_new = Tep_s+ts_deriv
        Tep_a_new = (Tep_c_new+Tep_s_new)/2
        # aging model
        Bc = self.Bc_func(Ic_rate)
        E = 31700-370.3*Ic_rate
        T = Tep_a_new+273.15
        Ah = (20/Bc/math.exp(-E/8.31/T))**(1/0.55)
        N1 = 3600*Ah/self.Cn
        dsoh = self.timestep*(abs(I_batt/2/N1/self.Cn))
        SOH_new = SOH-dsoh
        life_depleted_fraction = dsoh
        
        out_info = {'SoC': SOC_new, 'SoH': SOH_new,
                    'cell_OCV': Voc_new, 'cell_Vt': Vt_new, 'cell_V_3': V_3,
                    'current': I_batt, 'current_rate': Ic_rate, 'cell_power_out': power_out,
                    'battery_power': P_batt/1000, 'battery_temperature': Tep_a_new,
                    'life_depleted_fraction': life_depleted_fraction}
        paras_list_new = [SOC_new, SOH_new, Tep_c_new, Tep_s_new, Tep_a_new, Voc_new, V1_new, V2_new]
        done = fail
        return paras_list_new, life_depleted_fraction, I_batt, done, out_info
    
    def run_battery_cell2(self, P_batt, paras_list):
        fail = False
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a]
        SOC = paras_list[0]
        SOH = paras_list[1]
        Tep_c = paras_list[2]
        Tep_s = paras_list[3]
        Tep_a = paras_list[4]
        # battery electro model, Equivalent circuit model for whole pack
        # Battery efficiency, P_batt is of battery pack
        e_batt = (P_batt > 0)+(P_batt <= 0)*0.98
        # Battery internal resistance
        resistance = (P_batt > 0)*self.R_dis_func(SOC)+(P_batt <= 0)*self.R_chg_func(SOC)
        # Battery voltage
        V_batt = self.V_func(SOC)  # OCV
        # Battery pack current
        if V_batt**2 < 4*resistance*P_batt:
            I_batt = e_batt*V_batt/(2*resistance)
        else:
            I_batt = e_batt*(V_batt-math.sqrt(V_batt**2-4*resistance*P_batt))/(2*resistance)
        if I_batt > self.I_max:
            I_batt = self.I_max
            # fail = True
        # Suppose 100 cells are all connected in parallel
        # for a cell
        I_cell = I_batt / 11
        self.accumulated_Ah += abs(I_cell)*self.timestep/3600  # 1s
        Ic_rate = abs(I_cell/self.Cn)
        soc_deriv = self.timestep*(I_cell/3600/self.Cn)
        SOC_new = SOC-soc_deriv
        if SOC_new > 1:
            SOC_new = 1.0
            fail = True
        if SOC_new < 0.001:
            fail = True
            SOC_new = 0.001
        # thermal model
        cell_heat = resistance*(I_cell**2)      # H(t)
        tc_deriv = self.timestep*(((Tep_s-Tep_c)/self.Rc+cell_heat)/self.Cc)
        ts_deriv = self.timestep*(((Tep_c-Tep_s)/self.Rc+(Tep_a-Tep_s)/self.Ru)/self.Cs)
        Tep_c_new = Tep_c+tc_deriv
        Tep_s_new = Tep_s+ts_deriv
        Tep_a_new = (Tep_c_new+Tep_s_new)/2
        # need a limit for teperature
        
        # aging model
        Bc = self.Bc_func(Ic_rate)
        E = 31700-370.3*Ic_rate
        T = Tep_a_new+273.15
        Ah = (20/Bc/math.exp(-E/8.31/T))**(1/0.55)
        N1 = 3600*Ah/self.Cn
        dsoh = self.timestep*(abs(I_cell/2/N1/self.Cn))
        SOH_new = SOH-dsoh
        life_depleted_fraction = dsoh
        
        out_info = {'SoC': SOC_new, 'SoH': SOH_new, 'battery_current': I_batt, 'current_rate': Ic_rate,
                    'cell_current': I_cell, 'battery_power': P_batt/1000, 'battery_temperature': Tep_a_new,
                    'life_depleted_fraction': life_depleted_fraction}
        paras_list_new = [SOC_new, SOH_new, Tep_c_new, Tep_s_new, Tep_a_new]
        done = fail
        return paras_list_new, life_depleted_fraction, I_batt, done, out_info
        