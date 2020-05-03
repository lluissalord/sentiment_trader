import pandas as pd
import numpy as np

from gym import spaces
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions

# TODO: Normalize somehow the reward to be more standard between runs, independent on the data is processing
# TODO: Plot training info during training to be able to track it
class OwnStocksEnv(StocksEnv):

    def __init__(self, df, window_size, frame_bound, steps_per_episode, is_training, position_as_observation=True, constant_step=False, min_steps_per_episode=2, price_column='close', feature_columns=None, seed=None):

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
        self.max_possible_profit_df = self.max_possible_profit()

        self.trade_fee_bid_percent = 0.0 # 0.01  # unit
        self.trade_fee_ask_percent = 0.0 # 0.005  # unit

        self.position_as_observation = position_as_observation
        self.shape = (window_size, self.signal_features.shape[1] + int(position_as_observation))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.constant_step = constant_step

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

    def step(self, action):
        observation, reward, done, info = super().step(action)
        #print(observation, done, info)

        # TODO: Check if better use only final reward or step_rewards
        # reward = 0
        # if done:
        #     max_possible_revenue = self.max_possible_profit() - 1
        #     revenue = (info['total_profit'] - 1)
        #     if max_possible_revenue > 0:
        #         if revenue >= 0:
        #             reward = revenue / max_possible_revenue
        #         else:
        #             reward = 0
        #     elif max_possible_revenue < 0:
        #         reward = 0
        #     else:
        #         reward = revenue
        #     # TODO: Should this be modified?
        #     # info = dict(
        #     #     total_reward = self._total_reward,
        #     #     total_profit = self._total_profit,
        #     #     position = self._position.value
        #     # )

        # TODO: Check if should be this tick or the previous one
        max_possible_revenue = self.max_possible_profit_df.loc[self._current_tick, 'max_profit'] - 1
        revenue = (info['total_profit'] - 1)
        if max_possible_revenue > 0:
            if revenue >= 0:
                reward = revenue / max_possible_revenue
            else:
                reward = 0
        # TODO: Check if this case is not posible
        elif max_possible_revenue < 0:
            reward = 0
        else:
            reward = revenue

        # Normalize according to the number of steps of this episode
        reward /= (self._end_tick - self._start_tick)

        # Only for tracking of training
        # if done:    
        #     print(info['total_profit'] - 1, self.max_possible_profit() - 1)
            
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
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]): 
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
                max_possible_profit_df.loc[current_tick, 'max_profit'] = profit
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


def runTestEnv(env, select_action_func, iterations=None, use_steps=False, use_observation=False, use_model=False, isTFEnv=False, policy=None, seed=12345, **kwargs):
    if isTFEnv:
        TFEnv = env
        env = TFEnv.pyenv._envs[0]._gym_env
    env.seed(seed)
    
    if iterations is None:
        if env.is_training:
            iterations = int((env.frame_bound[1] - env.frame_bound[0]) / env.steps_per_episode)
        else:
            iterations = 1
    
    total_rewards = []
    total_profits = [] 

    for _ in range(iterations):
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
    
    print(f'Total rewards: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.3f} (mean ± std. dev. of {iterations} iterations)')
    print(f'Total profits: {(np.mean(total_profits) - 1):.2%} ± {np.std(total_profits):.3%} (mean ± std. dev. of {iterations} iterations)')

    return total_rewards, total_profits