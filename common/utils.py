# import inspect
# import functools
# from multiagent.environment import MultiAgentEnv
# import multiagent.scenarios as scenarios

# def store_args(method):
#     """Stores provided method args as instance attributes."""
#     argspec = inspect.getfullargspec(method)
#     defaults = {}
#     if argspec.defaults is not None:
#         defaults = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
#     if argspec.kwonlydefaults is not None:
#         defaults.update(argspec.kwonlydefaults)
#     arg_names = argspec.args[1:]
#
#     @functools.wraps(method)
#     def wrapper(*positional_args, **keyword_args):
#         self = positional_args[0]
#         # Get default arg values
#         args = defaults.copy()
#         # Add provided arg values
#         for name, value in zip(arg_names, positional_args[1:]):
#             args[name] = value
#         args.update(keyword_args)
#         self.__dict__.update(args)
#         return method(*positional_args, **keyword_args)
#     print('def store_args has been called.')
#     return wrapper

# def make_env(args):
#     # load scenario from script
#     scenario = scenarios.load(args.scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
#     # env = MultiAgentEnv(world)
#     # args.n_players = env.n  # 包含敌人的所有玩家个数
#     # args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
#     args.n_agents = env.n
#     args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
#     action_shape = []
#     for content in env.action_space:
#         # action_shape.append(content)
#         action_shape.append(content.n)  # ?
#     args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
#     args.high_action = 1
#     args.low_action = -1
#     speed_list = get_driving_cycle()
#     args.episode_steps = len(speed_list)                  # cycle length, be equal to args.episode_steps
#     return env, args

from env import Env
from utils2 import get_driving_cycle

def make_env2(args):

    env = Env()
    args.n_agents = env.agent_num
    args.obs_shape = [env.agents[i].obs_num for i in range(args.n_agents)]
    args.action_shape = [env.agents[i].action_num for i in range(args.n_agents)]
    args.high_action = 1
    args.low_action = -1
    speed_list = get_driving_cycle(cycle_name=args.scenario_name)
    args.episode_steps = len(speed_list)  # cycle length, be equal to args.episode_steps
    return env, args
