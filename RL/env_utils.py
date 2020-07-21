""" Utils for RL environments """

import numpy as np

def runAllTestEnv(all_envs, select_action_func, **kwargs):
    if type(all_envs) is list:
        all_envs = dict(zip([f'Env_{i}' for i in range(len(all_envs))], all_envs))

    if type(all_envs) is not dict:
        raise ValueError('all_envs should be dictionary of name and enviorment or a list of enviorments')
    else:
        for env_name, env in all_envs.items():
            print(f'Testing enviorment {env_name}:')
            runTestEnv(env, select_action_func, **kwargs)
            print('-'*50)


def runTestEnv(env, select_action_func, iterations=None, min_iterations=21, use_steps=False, use_observation=False, use_model=False, isTFEnv=False, policy=None, deterministic_policy=True, seed=12345, **kwargs):
    
    if isTFEnv:
        TFEnv = env
        env = TFEnv.pyenv._envs[0]._gym_env

    total_rewards = []
    total_profits = [] 
    total_revenue_ratio = []

    total_i = 0
    while total_i < min_iterations:
        env.seed(seed)
    
        if iterations is None:
            if env.is_training:
                iterations = int((env.frame_bound[1] - env.frame_bound[0]) / env.steps_per_episode)
            else:
                iterations = 1

        for i in range(iterations):
            if isTFEnv:
                time_step = TFEnv.reset()
                policy_state = policy.get_initial_state(TFEnv.batch_size)
            else:
                observation = env.reset()

            if use_model:
                done = False
                recent_observations = []
                recent_terminals = []
            step = 0
            while True:
                if isTFEnv:
                    action, done, time_step, policy_state = select_action_func(TFEnv=TFEnv, policy=policy, done=done, time_step=time_step, policy_state=policy_state, **kwargs)

                elif use_model:
                    action = select_action_func(observation, recent_observations, recent_terminals, done, **kwargs)

                    observation, _, done, _ = env.step(action)

                else:
                    if use_observation:
                        if use_steps:
                            action = select_action_func(observation=observation, step=step, **kwargs)
                        else:
                            action = select_action_func(observation=observation, **kwargs)
                    else:
                        if use_steps:
                            action = select_action_func(step=step, **kwargs)
                        else:
                            action = select_action_func(**kwargs)

                    observation, _, done, _ = env.step(action)
                
                if done:
                    break

                step += 1

            total_rewards.append(env._total_reward)
            total_profits.append(env._total_profit)
            total_revenue_ratio.append(env.calculate_revenue_ratio())
    
        if not env.is_training and deterministic_policy:
            break

        total_i += i + 1
        seed += 1
    
    print(f'Total rewards: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.3f} (mean ± std. dev. of {len(total_rewards)} iterations)')
    print(f'Total profits: {(np.mean(total_profits) - 1):.2%} ± {np.std(total_profits):.3%} (mean ± std. dev. of {len(total_profits)} iterations)')
    print(f'Total revenue ratio: {np.mean(total_revenue_ratio):.2%} ± {np.std(total_revenue_ratio):.3%} (mean ± std. dev. of {len(total_revenue_ratio)} iterations)')

    return total_rewards, total_profits, total_revenue_ratio