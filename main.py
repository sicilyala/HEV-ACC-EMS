# main program of training and evaluating
from runner import Runner
from arguments import get_args
from utils import make_env2
from evaluate import Evaluator
from data_process import data_process_new, draw_loss

if __name__ == '__main__':
    args = get_args()
    env, args = make_env2(args)
    print('max_episodes:', args.max_episodes)
    print('episode_steps:', args.episode_steps)
    print('n_agents:', args.n_agents)
    print('obs_shape:', args.obs_shape)
    print('action_shape:', args.action_shape)
    
    if args.evaluate:
        print("-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluating is finished!-----")
    else:
        print("-----Start training-----")
        runner = Runner(args, env)
        runner.run()
        print("-----Training is finished!-----")
        draw_loss(args.save_dir, args.scenario_name)
        data_process_new(args.save_dir, args.scenario_name, args.data_episode,
                         save_picture=True, show_max=True)
