import pandas as pd
import numpy as np

from gym import spaces
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions

REVENUE_REWARD = 1
PRICE_REWARD = 2

# TODO: Normalize somehow the reward to be more standard between runs, independent on the data is processing
# TODO: Plot training info during training to be able to track it
class RLStocksEnv(StocksEnv):

    def __init__(self, df, window_size, frame_bound, steps_per_episode, is_training, position_as_observation=True, constant_step=False, min_steps_per_episode=2, reward_type=REVENUE_REWARD, max_final_reward=100, max_step_reward=1, price_column='close', feature_columns=None, seed=None):

        self.price_column = price_column
        
        if feature_columns is None:
            self.feature_columns = list(df.columns)
        else:
            self.feature_columns = feature_columns

        super().__init__(df, window_size, frame_bound)
        
        if min_steps_per_episode <= 0:
            raise ValueError(f'min_steps_per_episode must be bigger than 0')

        self.seed(seed)
        self.steps_per_episode = steps_per_episode
        self.max_steps_per_episode = steps_per_episode
        self.min_steps_per_episode = min_steps_per_episode
        self.is_training = is_training

        self.trade_fee_bid_percent = 0.0 # 0.01  # unit
        self.trade_fee_ask_percent = 0.0 # 0.005  # unit

        self.max_possible_profit_df = self.max_possible_profit()

        self.position_as_observation = position_as_observation
        self.shape = (window_size, self.signal_features.shape[1] + int(position_as_observation))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.constant_step = constant_step

        self.max_final_reward = max_final_reward
        self.max_step_reward = max_step_reward

        self.reward_type = reward_type

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, self.price_column].to_numpy()[start:end]
        signal_features = self.df[self.feature_columns].to_numpy()[start:end]
        return prices, signal_features

    def reset(self, start_tick=None):
        if not self.constant_step:
            self.steps_per_episode = self.np_random.randint(self.max_steps_per_episode - self.min_steps_per_episode) + self.min_steps_per_episode

        if self.is_training:
            if start_tick is None:
                self._start_tick = min(
                    self.np_random.randint(self.frame_bound[1] - 1 - self.steps_per_episode) + self.window_size,
                    self.frame_bound[1] - self.steps_per_episode - 1
                )
            else:
                self._start_tick = start_tick
            self._end_tick = min(
                self._start_tick + self.steps_per_episode,
                self.frame_bound[1] - 1
            )
            
            self.max_possible_profit_df = self.max_possible_profit()

        return super().reset()

    def calculate_revenue_ratio(self):
        # TODO: Check if should be this tick or the previous one
        max_possible_revenue = self.max_possible_profit_df.loc[self._current_tick, 'max_profit'] - 1
        revenue = self._total_profit - 1
        if max_possible_revenue > 0 and revenue >= 0:
            revenue_ratio = revenue / max_possible_revenue
        # TODO: Check if it is good to have this behaviour or if it will impact too much on not buy/sell in a lot of moments
        elif max_possible_revenue == 0 and revenue == 0:
            revenue_ratio = 1
        # TODO: Check if use this or better to set to 0
        else:
            # revenue_ratio = revenue
            revenue_ratio = 0
        return revenue_ratio

    def step(self, action):
        observation, reward, done, info = super().step(action)

        if self.reward_type == REVENUE_REWARD:
            revenue_ratio = self.calculate_revenue_ratio()

            if done:
                reward = self.max_final_reward * revenue_ratio
            else:
                reward = self.max_step_reward * revenue_ratio

                # Normalize according to the number of steps of this episode
                reward /= (self._end_tick - self._start_tick)
        elif self.reward_type == PRICE_REWARD:
            reward /= self.prices[self._last_trade_tick]
            
        return observation, reward, done, info

    def _get_observation(self):
        features = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        
        if self.position_as_observation:
            positions = np.expand_dims(
                np.array(
                    list(
                        map(
                            lambda position: position.value if position is not None else 0,
                            self._position_history[-self.window_size:]
                        )
                    )
                ),
                axis=1
            )
            return np.append(
                features,
                positions,
                axis=1
            )
        else:
            return features

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        max_possible_profit_df = pd.DataFrame(index=range(self._start_tick, self._end_tick + 1), columns=['max_profit'])
        max_possible_profit_df.loc[current_tick, 'max_profit'] = profit

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]): 
                    current_tick += 1
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    
                    shares = (profit * (1 - self.trade_fee_ask_percent)) / self.prices[last_trade_tick]
                    temp_profit = (shares * (1 - self.trade_fee_bid_percent)) * self.prices[current_tick]
                    max_possible_profit_df.loc[current_tick, 'max_profit'] = temp_profit

                    current_tick += 1
                profit = temp_profit
            last_trade_tick = current_tick - 1

        max_possible_profit_df = max_possible_profit_df.fillna(method='ffill')

        return max_possible_profit_df


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