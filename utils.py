import numpy as np
import collections
import os
import time

from absl import logging

from tensorflow import equal as tf_equal
from tensorflow import add as tf_add
from tensorflow.compat.v2 import summary

from gym import spaces
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy

from tf_agents.policies import policy_saver

from own_stock_env import OwnStocksEnv, REVENUE_REWARD, PRICE_REWARD

def generateSplitEnvs(
    train_df,
    valid_df,
    test_df,
    window_size,
    steps_per_episode,
    feature_columns,
    reward_type=REVENUE_REWARD,
    max_final_reward=1,
    max_step_reward=0,
    num_parallel_environments=1,
    position_as_observation=True,
    constant_step=False,
    is_training=True,
    seed=12345,
):

    eval_env = OwnStocksEnv(
        df=valid_df,
        window_size=window_size,
        frame_bound=(window_size, len(valid_df)),
        steps_per_episode=steps_per_episode,
        is_training=is_training,
        constant_step=constant_step,
        feature_columns=feature_columns,
        position_as_observation=position_as_observation,
        reward_type=reward_type,
        max_final_reward=max_final_reward,
        max_step_reward=max_step_reward,
    )
    eval_env.seed(seed)
    eval_env.reset()

    test_env = OwnStocksEnv(
        df=test_df,
        window_size=window_size,
        frame_bound=(window_size, len(test_df)),
        steps_per_episode=steps_per_episode,
        is_training=is_training,
        constant_step=constant_step,
        feature_columns=feature_columns,
        position_as_observation=position_as_observation,
        reward_type=reward_type,
        max_final_reward=max_final_reward,
        max_step_reward=max_step_reward,
    )
    test_env.seed(seed)
    test_env.reset()

    # Otherwise raise error on evaluating ChosenActionHistogram metric
    spec_dtype_map = {spaces.Discrete: np.int32}

    tf_parallel_envs = []
    for i in range(num_parallel_environments):
        train_env = OwnStocksEnv(
            df=train_df,
            window_size=window_size,
            frame_bound=(window_size, len(train_df)),
            steps_per_episode=steps_per_episode,
            is_training=True,
            constant_step=constant_step,
            feature_columns=feature_columns,
            position_as_observation=position_as_observation,
            reward_type=reward_type,
            max_final_reward=max_final_reward,
            max_step_reward=max_step_reward,
        )
        train_env.seed(seed + i)
        train_env.reset()
        tf_parallel_envs.append(
            GymWrapper(train_env, spec_dtype_map=spec_dtype_map)
        )

    # TODO: Implement Parallel Environment (need tf_agents.system.multiprocessing.enable_interactive_mode() added in github last updates)
    if num_parallel_environments != 1:
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(tf_parallel_envs))
    else:
        tf_env = tf_py_environment.TFPyEnvironment(tf_parallel_envs[0])

    eval_tf_env = tf_py_environment.TFPyEnvironment(GymWrapper(eval_env, spec_dtype_map=spec_dtype_map))
    test_tf_env = tf_py_environment.TFPyEnvironment(GymWrapper(test_env, spec_dtype_map=spec_dtype_map))

    return tf_env, eval_tf_env, test_tf_env


class AgentEarlyStopping():
  def __init__(self,
               monitor='AverageReturn',
               min_delta=0,
               patience=0,
               patience_after_change=0,
               verbose=0,
               mode='max',
               baseline=None):
    """Initialize an AgentEarlyStopping.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of iterations with no improvement
            after which training will be stopped.
        patience_after_change: Number of iterations after change
             on monitor with no improvement after which training
             will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
    """
    # super(AgentEarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    if patience_after_change > patience:
        self.patience_after_change = patience
    else:
        self.patience_after_change = patience_after_change
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      elif 'return' in self.monitor.lower():
        self.monitor_op = np.less
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

    self.reset()

  def reset(self):
    # Allow instances to be re-used
    self.wait = 0
    self.wait_after_change = 0
    self.stopped_step = 0
    self.stop_training = False
    self.monitor_changed = False
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  # TODO: Calculate a EWMA with alpha = 0.999 and calculate max buffer with length = (log 0.01) / (log 0.999) (being 0.01 minimum weight)
  def __call__(self, computed_metrics, global_step):
    current = self.get_monitor_value(computed_metrics)
    if current is None:
      return
    if not tf_equal(current, self.best) and not np.isinf(self.best):
        self.monitor_changed = True
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      self.wait_after_change = 0
    else:
      self.wait += 1
      if self.monitor_changed:
          self.wait_after_change += 1
      if self.wait >= self.patience or self.wait_after_change >= self.patience_after_change:
        self.stopped_step = global_step
        self.stop_training = True
        logging.info('Global step %05d: early stopping' % (self.stopped_step + 1))

  def get_monitor_value(self, computed_metrics):
    computed_metrics = computed_metrics or {}
    monitor_value = computed_metrics.get(self.monitor).numpy()
    if monitor_value is None:
      logging.warning('Agent early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(computed_metrics.keys())))
    return monitor_value


def evaluate(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, num_eval_seeds, global_step=None, eval_summary_writer=None, summary_prefix='Metrics', seed=12345):
    all_results = []
    for i in range(num_eval_seeds):
        for env in eval_tf_env.envs:
            env.seed(seed + i)
        # One final eval before exiting.
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
        )
        all_results.append(results)

    mean_results = collections.OrderedDict(results)
    if num_eval_seeds > 1:
        for metric in mean_results:
            metric_sum = 0
            for result in all_results:
                metric_sum = tf_add(metric_sum, result[metric])
            mean_results[metric] = metric_sum / len(all_results)
    if global_step and eval_summary_writer:
        with eval_summary_writer.as_default():
            for metric, value in mean_results.items():
                tag = common.join_scope(summary_prefix, metric)
                summary.scalar(name=tag, data=value, step=global_step)

    log = ['{0} = {1}'.format(metric, value) for metric, value in mean_results.items()]
    logging.info('%s \n\t\t %s','', '\n\t\t '.join(log))

    return mean_results


def train_eval(tf_agent, num_iterations, batch_size, tf_env, eval_tf_env, train_metrics, step_metrics, eval_metrics, global_step, replay_buffer_capacity, num_parallel_environments, collect_per_iteration, train_steps_per_iteration, train_dir, saved_model_dir, eval_summary_writer, num_eval_episodes, num_eval_seeds=1, eval_metrics_callback=None, train_sequence_length=1, initial_collect_steps=1000, log_interval=100, eval_interval=400, policy_checkpoint_interval=400, train_checkpoint_interval=1200, rb_checkpoint_interval=2000, train_model=True, use_tf_functions=True, eval_early_stopping=False, seed=12345):

    tf_agent.initialize()
    agent_name = tf_agent.__dict__['_name']

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=num_parallel_environments, # batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    if train_model:
      if agent_name in ['dqn_agent']:
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=collect_per_iteration)
      elif agent_name in ['ppo_agent']:
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_per_iteration)
      else:
          raise NotImplementedError(f'{agent_name} agent not yet implemented')

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    policy_checkpointer.initialize_or_restore() # TODO: To be tested
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    if train_model:

      # TODO: should they use autograph=False?? as in tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py
      if use_tf_functions:
        # To speed up collect use common.function.
        collect_driver.run = common.function(collect_driver.run) 
        tf_agent.train = common.function(tf_agent.train)

      # Only run Replay buffer initialization if using one of the following agents
      if agent_name in ['dqn_agent']:
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())

        # Collect initial replay data.
        logging.info(
            'Initializing replay buffer by collecting experience for %d steps with '
            'a random policy.', initial_collect_steps)
        dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=initial_collect_steps).run()

      logging.info(
          f'Initial eval metric'
      )
      results = evaluate(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, num_eval_seeds, global_step, eval_summary_writer, summary_prefix='Metrics', seed=seed)

      if eval_early_stopping and not isinstance(eval_metrics_callback, AgentEarlyStopping):
          raise ValueError('Cannot set eval_early_stopping without eval_metric_callback being Agent Early Stopping instance')

      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())

      time_step = None
      policy_state = collect_policy.get_initial_state(tf_env.batch_size)

      timed_at_step = global_step.numpy()
      collect_time = 0
      train_time = 0
      summary_time = 0

      if agent_name in ['dqn_agent']:
        # Dataset generates trajectories with shape [Bx2x...]
        logging.info(
            f'Dataset generates trajectories'
        )
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            # single_deterministic_pass=True,
            num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)

        def train_step():
          experience, _ = next(iterator)
          return tf_agent.train(experience)
      elif agent_name in ['ppo_agent']:
        def train_step():
          trajectories = replay_buffer.gather_all()
          return tf_agent.train(experience=trajectories)
      else:
        raise NotImplementedError(f'{agent_name} agent not yet implemented')

      if use_tf_functions:
        train_step = common.function(train_step)

      logging.info(
            f'Starting training...'
      )
      for _ in range(num_iterations):
        start_time = time.time()
        if agent_name in ['dqn_agent']:
          time_step, policy_state = collect_driver.run(
              time_step=time_step,
              policy_state=policy_state,
          )
        elif agent_name in ['ppo_agent']:
          collect_driver.run()
        else:
          raise NotImplementedError(f'{agent_name} agent not yet implemented')
        
        collect_time += time.time() - start_time

        start_time = time.time()
        for _ in range(train_steps_per_iteration):
          train_loss = train_step()
        train_time += time.time() - start_time

        start_time = time.time()
        for train_metric in train_metrics:
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=step_metrics)
        summary_time += time.time() - start_time

        if global_step.numpy() % log_interval == 0:
          logging.info('step = %d, loss = %f', global_step.numpy(),
                      train_loss.loss)
          steps_per_sec = (global_step.numpy() - timed_at_step) / (train_time + collect_time + summary_time)
          logging.info('%.3f steps/sec', steps_per_sec)
          logging.info('collect_time = %.3f, train_time = %.3f, summary_time = %.3f', collect_time,
                     train_time, summary_time)
          summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step)
          timed_at_step = global_step.numpy()
          collect_time = 0
          train_time = 0
          summary_time = 0

        if global_step.numpy() % train_checkpoint_interval == 0:
          start_time = time.time()
          train_checkpointer.save(global_step=global_step.numpy())
          logging.info(
            f'Saving Train lasts: {time.time() - start_time:.3f} s'
          )

        if global_step.numpy() % policy_checkpoint_interval == 0:
          start_time = time.time()
          policy_checkpointer.save(global_step=global_step.numpy())
          saved_model_path = os.path.join(
              saved_model_dir, 'policy_' + ('%d' % global_step.numpy()).zfill(9))
          saved_model.save(saved_model_path)
          logging.info(
            f'Saving Policy lasts: {time.time() - start_time:.3f} s'
          )

        if global_step.numpy() % rb_checkpoint_interval == 0:
          start_time = time.time()
          rb_checkpointer.save(global_step=global_step.numpy())
          logging.info(
            f'Saving Replay Buffer lasts: {time.time() - start_time:.3f} s'
          )

        if global_step.numpy() % eval_interval == 0:
          start_time = time.time()
          results = evaluate(eval_metrics, eval_tf_env, eval_policy, num_eval_episodes, num_eval_seeds, global_step, eval_summary_writer, summary_prefix='Metrics', seed=seed)
          if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
          logging.info(
            f'Calculate Evaluation lasts {time.time() - start_time:.3f} s'
          )

          if eval_early_stopping and eval_metrics_callback.stop_training:
              logging.info(
                  f'Training stopped due to Agent Early Stopping at step: {global_step.numpy()}'
              )
              logging.info(
                  f'Best {eval_metrics_callback.monitor} was {eval_metrics_callback.best:.5f} at step {eval_metrics_callback.stopped_step}'
              )
              break