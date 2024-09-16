#  HEV-ACC-EMS: Ecological Driving Framework of Hybrid Electric Vehicle Based on Heterogeneous Multi-Agent Deep Reinforcement Learning.
## Overview

The original implementation of **Ecological Driving Framework of Hybrid Electric Vehicle Based on Heterogeneous Multi-Agent Deep Reinforcement Learning.**


## Abstract

Hybrid electric vehicles (HEVs) have great potential to be discovered in terms of energy saving and emission reduction, and ecological driving provides theoretical guidance for giving full play to their advantages in real traffic scenarios. In order to implement an ecological driving strategy with the lowest cost throughout the life cycle in a car-following scenario, the safety and comfort, fuel economy, and battery health need to be considered, which is a complex nonlinear and multiobjective coupled optimization task. Therefore, a novel multi-agent deep deterministic policy gradient (MADDPG) based framework with two heterogeneous agents to optimize adaptive cruise control (ACC) and energy management strategy (EMS), respectively, is proposed, thereby decoupling optimization objectives of different domains. Because of the asynchronous of multi-agents, different learning rate schedules are analyzed to coordinate the learning process to optimize training results; an improvement on the prioritized experience replay (PER) technique is proposed, which improves the optimization performance of the original MADDPG method by more than 10%. Simulations under mixed driving cycles show that, on the premise of ensuring car-following performance, the overall driving cost, including fuel consumption and battery health degradation of the MADDPG-based method, can reach93.88% of that of DP, and the proposed algorithm has good adaptability to different driving conditions.


## Data

1. **Driving Cycles can be found [here](https://github.com/sicilyala/project-data/tree/main/standard_driving_cycles).**

2. **Power system data can be found [here](https://github.com/sicilyala/project-data/tree/main/HEV_data).**


## Citation
**BibTex**
```
@article{10130313,
title = {Ecological Driving Framework of Hybrid Electric Vehicle Based on Heterogeneous Multi-Agent Deep Reinforcement Learning},
journal = {IEEE Transactions on Transportation Electrification},
volume = {10},
pages = {392-406},
year = {2024},
doi = {10.1109/TTE.2023.3278350},
url = {https://ieeexplore.ieee.org/abstract/document/10130313},
author = {Peng, Jiankun and Chen, Weiqi and Fan, Yi and He, Hongwen and Wei, Zhongbao and Ma, Chunye},
index-terms = {Adaptive cruise control(ACC), battery health, deep deterministic policy gradient, ecological driving, energy management, heterogeneous multi-agent.},
}
```
