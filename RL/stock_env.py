""" Stock environment class based on Gym_anytrading environment """

import pandas as pd
import numpy as np

from gym import spaces
from gym_anytrading.envs import TradingEnv, StocksEnv, Actions, Positions

REVENUE_REWARD = 1
PRICE_REWARD = 2

# TODO: Normalize somehow the reward to be more standard between runs, independent on the data is processing
# TODO: Plot training info during training to be able to track it
class RLStocksEnv(StocksEnv):
    """ Stock environment class based on Gym_anytrading environment """

    def __init__(self, df, window_size, frame_bound, steps_per_episode, is_training, position_as_observation=True, constant_step=False, min_steps_per_episode=2, reward_type=REVENUE_REWARD, max_final_reward=100, max_step_reward=1, price_column='close', feature_columns=None, trade_fee_bid_percent=0, trade_fee_ask_percent=0, seed=None):

        # Initialize members of the class with default values
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

        self.trade_fee_bid_percent = trade_fee_bid_percent # 0.01  # unit
        self.trade_fee_ask_percent = trade_fee_ask_percent # 0.005  # unit

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
        """ Reset environment """
        # For non-constant step, set a random steps_per_episode
        if not self.constant_step:
            self.steps_per_episode = self.np_random.randint(self.max_steps_per_episode - self.min_steps_per_episode) + self.min_steps_per_episode

        # In case of being training, on each reset/episode, the start tick can be provided, otherwise it is set randomly
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
            
            # Calculate maximum possible profit dataframe on each reset/episode, depending on start and steps per episode defined
            self.max_possible_profit_df = self.max_possible_profit()

        return super().reset()

    def calculate_revenue_ratio(self):
        """ Calculate revenue ratio based on current revenue (total profit - 1) and max_possible_revenue at current step """

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
        """ Perform step with provided action """

        # Perform step based on StocksEnv parent class
        # Reward from parent class is calculated based on price difference (between Short and Long)
        observation, reward, done, info = super().step(action)

        # Reward can be calculated based on revenue with final reward or step reward
        if self.reward_type == REVENUE_REWARD:
            revenue_ratio = self.calculate_revenue_ratio()

            if done:
                reward = self.max_final_reward * revenue_ratio
            else:
                reward = self.max_step_reward * revenue_ratio

                # Normalize according to the number of steps of this episode
                reward /= (self._end_tick - self._start_tick)

        # Reward can be based on price (as parent class), but normalized by the Short price
        elif self.reward_type == PRICE_REWARD:
            reward /= self.prices[self._last_trade_tick]
            
        return observation, reward, done, info

    def _get_observation(self):
        """ Return observation of the current tick """
        features = self.signal_features[(self._current_tick-self.window_size):self._current_tick]
        
        # Add the current position (Short or Long) to the observation data
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
        """ Calculate maximum possible profit given the current parameters (prices, start tick and end tick) """

        # Initialize variables
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        # Initialize DataFrame
        max_possible_profit_df = pd.DataFrame(index=range(self._start_tick, self._end_tick + 1), columns=['max_profit'])
        max_possible_profit_df.loc[current_tick, 'max_profit'] = profit

        while current_tick <= self._end_tick:
            # Increase ticks till there is oportunity to go Long or Short (change of direction)
            # In that case, simulate it sells/buy at the best moment
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]): 
                    current_tick += 1
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    
                    # Calculate profits when selling and set value on Dataframe
                    shares = (profit * (1 - self.trade_fee_ask_percent)) / self.prices[last_trade_tick]
                    temp_profit = (shares * (1 - self.trade_fee_bid_percent)) * self.prices[current_tick]
                    max_possible_profit_df.loc[current_tick, 'max_profit'] = temp_profit

                    current_tick += 1
                profit = temp_profit
            last_trade_tick = current_tick - 1

        # Fill NaN values with previous profit
        max_possible_profit_df = max_possible_profit_df.fillna(method='ffill')

        return max_possible_profit_df
