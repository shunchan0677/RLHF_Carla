#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import logging
import os
from os import path as osp
import sys
import time
from multiprocessing import Process, Queue
from pref_db import Segment

import queue

import cloudpickle
#import easy_tf_log
from a2c import logger
from a2c.a2c.a2c import learn
from a2c.a2c.policies import CnnPolicy, MlpPolicy
from a2c.common import set_global_seeds
from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from params import parse_args, PREFS_VAL_FRACTION
from pref_db import PrefDB, PrefBuffer
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from reward_predictor_core_network import net_cnn, net_moving_dot_features
from utils import VideoRenderer, get_port_range, make_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # filter out INFO messages


from absl import app
from absl import flags
from absl import logging

import functools
import gin
import numpy as np
import os
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
#tf.compat.v1.enable_resource_variables()
import time
import collections

import gym
import gym_carla

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from interp_e2e_driving.agents.latent_sac import latent_sac_agent
from interp_e2e_driving.environments import filter_observation_wrapper
from interp_e2e_driving.networks import sequential_latent_network
from interp_e2e_driving.utils import gif_utils


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name used for naming the output directory.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS

class FilterObservationWrapperRewards(wrappers.PyEnvironmentBaseWrapper):
  """Environment wrapper to filter observation channels."""

  def __init__(self, gym_env, input_channels,reward_predictor,gen_segments = None,
                 seg_pipe = None):
    super(FilterObservationWrapperRewards, self).__init__(gym_env)
    self.input_channels = input_channels

    observation_spaces = collections.OrderedDict()
    for channel in self.input_channels:
      observation_spaces[channel] = self._env.observation_space[channel]
    self.observation_space = gym.spaces.Dict(observation_spaces)
    self.reward_predictor = reward_predictor

    self.episode_vid_queue = None
    self.gen_segments = gen_segments
    self.segment = Segment()
    self.seg_pipe = seg_pipe
    self.nsteps = 1

    if(self.reward_predictor):
       print("reward_predictor")

  def _modify_observation(self, observation):
    observations = collections.OrderedDict()
    for channel in self.input_channels:
      observations[channel] = observation[channel]
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)

    if(self.reward_predictor):
       nenvs = 1
       nstack = 4
       nsteps = self.nsteps
       nh, nw, nc = observation["birdeye"].shape
       obs =  np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)
       obs[:, :, :, -3:] = observation["birdeye"][:, :, :]
       mb_obs = []
       mb_obs.append(np.copy(obs))
       mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
       mb_obs_allenvs = mb_obs.reshape(nenvs * nsteps, 64, 64, 3*4)

       rewards_allenvs = self.reward_predictor.raw_rewards(mb_obs_allenvs)[0]
       #rewards_allenvs = self.reward_predictor.reward(mb_obs_allenvs)

       #mb_rewards = rewards_allenvs.reshape(nenvs, nsteps)
       #mb_rewards = mb_rewards.flatten()
       if self.gen_segments:
            self.update_segment_buffer([mb_obs], [[reward]], [[done]])

       # Save frames for episode rendering
       #if self.episode_vid_queue is not None:
       #     self.update_episode_frame_buffer(mb_obs, done)

       reward = rewards_allenvs[0]



    observation = self._modify_observation(observation)


    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)
  

  def update_segment_buffer(self, mb_obs, mb_rewards, mb_dones):
        # Segments are only generated from the first worker.
        # Empirically, this seems to work fine.
        e0_obs = (mb_obs[0] * 255).astype(np.uint8)
        e0_rew = mb_rewards[0]
        e0_dones = mb_dones[0]

        for step in range(self.nsteps):
            self.segment.append(np.copy(e0_obs[step]), np.copy(e0_rew[step]))
            if len(self.segment) == 25 or e0_dones[step]:
                while len(self.segment) < 25:
                    # Pad to 25 steps long so that all segments in the batch
                    # have the same length.
                    # Note that the reward predictor needs the full frame
                    # stack, so we send all frames.
                    self.segment.append(e0_obs[step], 0)
                self.segment.finalise()
                try:
                    self.seg_pipe.put(self.segment, block=False)
                except queue.Full:
                    # If the preference interface has a backlog of segments
                    # to deal with, don't stop training the agents. Just drop
                    # the segment and keep on going.
                    pass
                self.segment = Segment()

  def update_episode_frame_buffer(self, mb_obs, mb_dones):
        e0_obs = (mb_obs[0] * 255).astype(np.uint8)
        e0_dones = mb_dones[0]
        for step in range(self.nsteps):
            # Here we only need to send the last frame (the most recent one)
            # from the 4-frame stack, because we're just showing output to
            # the user.
            self.episode_frames.append(e0_obs[step, :, :, -3:])
            if e0_dones[step]:
                self.episode_vid_queue.put(self.episode_frames)
                self.episode_frames = []


@gin.configurable
def load_carla_env(
  env_name='carla-v0',
  discount=1.0,
  number_of_vehicles=100,
  number_of_walkers=0,
  display_size=256,
  max_past_step=1,
  dt=0.1,
  discrete=False,
  discrete_acc=[-3.0, 0.0, 3.0],
  discrete_steer=[-0.2, 0.0, 0.2],
  continuous_accel_range=[-3.0, 3.0],
  continuous_steer_range=[-0.3, 0.3],
  ego_vehicle_filter='vehicle.lincoln*',
  port=2000,
  town='Town03',
  task_mode='random',
  max_time_episode=500,
  max_waypt=12,
  obs_range=32,
  lidar_bin=0.5,
  d_behind=12,
  out_lane_thres=2.0,
  desired_speed=8,
  max_ego_spawn_times=200,
  display_route=True,
  pixor_size=64,
  pixor=False,
  obs_channels=None,
  action_repeat=1,
  reward_predictor=None,
  gen_segments=None,
  seg_pipe=None
  ):
  """Loads train and eval environments."""
  env_params = {
    'number_of_vehicles': number_of_vehicles,
    'number_of_walkers': number_of_walkers,
    'display_size': display_size,  # screen size of bird-eye render
    'max_past_step': max_past_step,  # the number of past steps to draw
    'dt': dt,  # time interval between two frames
    'discrete': discrete,  # whether to use discrete control space
    'discrete_acc': discrete_acc,  # discrete value of accelerations
    'discrete_steer': discrete_steer,  # discrete value of steering angles
    'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
    'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
    'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
    'port': port,  # connection port
    'town': town,  # which town to simulate
    'task_mode': task_mode,  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': max_time_episode,  # maximum timesteps per episode
    'max_waypt': max_waypt,  # maximum number of waypoints
    'obs_range': obs_range,  # observation range (meter)
    'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
    'd_behind': d_behind,  # distance behind the ego vehicle (meter)
    'out_lane_thres': out_lane_thres,  # threshold for out of lane
    'desired_speed': desired_speed,  # desired speed (m/s)
    'max_ego_spawn_times': max_ego_spawn_times,  # maximum times to spawn ego vehicle
    'display_route': display_route,  # whether to render the desired route
    'pixor_size': pixor_size,  # size of the pixor labels
    'pixor': pixor,  # whether to output PIXOR observation
  }

  gym_spec = gym.spec(env_name)
  gym_env = gym_spec.make(params=env_params)

  if obs_channels:
    gym_env = FilterObservationWrapperRewards(gym_env, obs_channels,reward_predictor,gen_segments,seg_pipe)
    #gym_env_eval = FilterObservationWrapperRewards(gym_env, obs_channels,None,None,None)

  py_env = gym_wrapper.GymWrapper(
    gym_env,
    discount=discount,
    auto_reset=True,
  )

  #eval_py_env = gym_wrapper.GymWrapper(
  #  gym_env_eval,
  #  discount=discount,
  #  auto_reset=True,
  #)

  eval_py_env = py_env


  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)

  return py_env, eval_py_env


def compute_summaries(metrics,
                      environment,
                      policy,
                      train_step=None,
                      summary_writer=None,
                      num_episodes=1,
                      num_episodes_to_render=1,
                      model_net=None,
                      fps=10,
                      image_keys=None):
  for metric in metrics:
    metric.reset()

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)

  if num_episodes_to_render:
    images = [[time_step.observation]]  # now images contain dictionary of images
    latents = [[policy_state[1]]]
  else:
    images = []
    latents = []

  # Get input images and latent states
  episode = 0
  while episode < num_episodes:
    action_step = policy.action(time_step, policy_state)
    next_time_step = environment.step(action_step.action)
    policy_state = action_step.state

    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    for observer in metrics:
      observer(traj)

    if episode < num_episodes_to_render:
      if traj.is_boundary():
        images.append([])
        latents.append([])
      images[-1].append(next_time_step.observation)
      latents[-1].append(policy_state[1])

    if traj.is_last():
      episode += 1
      policy_state = policy.get_initial_state(environment.batch_size)

    time_step = next_time_step

  # Summarize scalars to tensorboard
  if train_step and summary_writer:
    with summary_writer.as_default():
      for m in metrics:
        tag = m.name
        tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)

  # Concat input images of different episodes and generate reconstructed images.
  # Shape of images is [[images in episode as timesteps]].
  if type(images[0][0]) is collections.OrderedDict:
    images = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=True)
  else:
    images = pad_and_concatenate_videos(images, image_keys=image_keys, is_dict=False)
  images = tf.image.convert_image_dtype([images], tf.uint8, saturate=True)
  images = tf.squeeze(images, axis=2)
  
  reconstruct_images = get_latent_reconstruction_videos(latents, model_net)
  reconstruct_images = tf.image.convert_image_dtype([reconstruct_images], tf.uint8, saturate=True)

  # Need to avoid eager here to avoid rasing error
  gif_summary = common.function(gif_utils.gif_summary_v2)

  # Summarize to tensorboard
  gif_summary('ObservationVideoEvalPolicy', images, 1, fps)
  gif_summary('ReconstructedVideoEvalPolicy', reconstruct_images, 1, fps)


def pad_and_concatenate_videos(videos, image_keys, is_dict=False):
  max_episode_length = max([len(video) for video in videos])
  if is_dict:
    # videos = [[tf.concat(list(dict_obs.values()), axis=2) for dict_obs in video] for video in videos]
    videos = [[tf.concat([dict_obs[key] for key in image_keys], axis=2) for dict_obs in video] for video in videos]
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=1) for frames in zip(*videos)]
  return videos


def get_latent_reconstruction_videos(latents, model_net):
  videos = []
  for latent_eps in latents:
    videos.append([])
    for latent in latent_eps:
      videos[-1].append(model_net.reconstruct(latent)[0])

  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    #　video contains [dict_obs of timesteps]
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  #　frames is [(each episodes obs at timestep t)]
  videos = [tf.concat(frames, axis=0) for frames in zip(*videos)]
  return videos


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


class Preprocessing_Layer(tf.keras.layers.Layer):
  """Preprocessing layers for multiple source inputs."""

  def __init__(self, base_depth, feature_size, name=None):
    super(Preprocessing_Layer, self).__init__(name=name)
    self.base_depth = base_depth
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

  def __call__(self, image, training=None):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

  def get_config(self):
    config = {
      'base_depth':self.base_depth, 
      'feature_size':self.feature_size}
    return config


@gin.configurable
def train_eval(
    root_dir,
    experiment_name,  # experiment name
    env_name='carla-v0',
    agent_name='sac',  # agent's name
    num_iterations=int(1e7),
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    model_network_ctor_type='non-hierarchical',  # model net
    input_names=['camera', 'lidar'],  # names for inputs
    mask_names=['birdeye'],  # names for masks
    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=int(1e5),
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    initial_model_train_steps=100000,  # initial model training
    batch_size=256,
    model_batch_size=32,  # model training batch size
    sequence_length=4,  # number of timesteps to train model
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    model_learning_rate=1e-4,  # learning rate for model training
    td_errors_loss_fn=tf.losses.mean_squared_error,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for summaries and logging
    num_images_per_summary=1,  # images for each summary
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    gpu_allow_growth=True,  # GPU memory growth
    gpu_memory_limit=None,  # GPU memory limit
    action_repeat=1,
    py_env=None, 
    eval_py_env=None,
    seg_pipe=None,
    start_policy_training_pipe=None,
    episode_vid_queue=None,
    reward_predictor=None,
    ckpt_save_dir=None,
    gen_segments=None):  # Name of single observation channel, ['camera', 'lidar', 'birdeye']
  # Setup GPU
  #tf.compat.v1.enable_resource_variables()


  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_allow_growth:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpu_memory_limit:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  # Get train and eval direction
  root_dir = os.path.expanduser(root_dir)
  root_dir = os.path.join(root_dir, env_name, experiment_name)

  # Get summary writers
  summary_writer = tf.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  # Eval metrics
  eval_metrics = [
      tf_metrics.AverageReturnMetric(
        name='AverageReturnEvalPolicy', buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(
        name='AverageEpisodeLengthEvalPolicy',
        buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  

  # Whether to record for summary
  with tf.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    # Create Carla environment
    #if agent_name == 'latent_sac':

    
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    fps = int(np.round(1.0 / (py_env.dt * action_repeat)))

    # Specs
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    ## Make tf agent
    if agent_name == 'latent_sac':
      # Get model network for latent sac
      if model_network_ctor_type == 'hierarchical':
        model_network_ctor = sequential_latent_network.SequentialLatentModelHierarchical
      model_net = model_network_ctor(input_names, input_names+mask_names)

      # Get the latent spec
      latent_size = model_net.latent_size
      latent_observation_spec = tensor_spec.TensorSpec((latent_size,), dtype=tf.float32)
      latent_time_step_spec = ts.time_step_spec(observation_spec=latent_observation_spec)

      # Get actor and critic net
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          latent_observation_spec,
          action_spec,
          fc_layer_params=actor_fc_layers,
          continuous_projection_net=normal_projection_net)
      critic_net = critic_network.CriticNetwork(
          (latent_observation_spec, action_spec),
          observation_fc_layer_params=critic_obs_fc_layers,
          action_fc_layer_params=critic_action_fc_layers,
          joint_fc_layer_params=critic_joint_fc_layers)

      # Build the inner SAC agent based on latent space
      inner_agent = sac_agent.SacAgent(
          latent_time_step_spec,
          action_spec,
          actor_network=actor_net,
          critic_network=critic_net,
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=actor_learning_rate),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=critic_learning_rate),
          alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=alpha_learning_rate),
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          td_errors_loss_fn=td_errors_loss_fn,
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          gradient_clipping=gradient_clipping,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step)
      inner_agent.initialize()

      # Build the latent sac agent
      tf_agent = latent_sac_agent.LatentSACAgent(
          time_step_spec,
          action_spec,
          inner_agent=inner_agent,
          model_network=model_net,
          model_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=model_learning_rate),
          model_batch_size=model_batch_size,
          num_images_per_summary=num_images_per_summary,
          sequence_length=sequence_length,
          gradient_clipping=gradient_clipping,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step,
          fps=fps)
      tf_agent.initialize()

    # Get replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,  # No parallel environments
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    # Train metrics
    env_steps = tf_metrics.EnvironmentSteps()
    average_return = tf_metrics.AverageReturnMetric(
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size)
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        env_steps,
        average_return,
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size),
    ]

    # Get policies
    # eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    eval_policy = tf_agent.policy
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec, action_spec)
    collect_policy = tf_agent.collect_policy

    # Checkpointers
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=None)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step,
        max_to_keep=None)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    # Collect driver
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration)
    

    # Optimize the performance by using tf functions
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    if (env_steps.result() == 0 or replay_buffer.num_frames() == 0):
      logging.info(
          'Initializing replay buffer by collecting experience for %d steps'
          'with a random policy.', initial_collect_steps)
      initial_collect_driver.run()

    compute_summaries(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        train_step=global_step,
        summary_writer=summary_writer,
        num_episodes=1,
        num_episodes_to_render=1,
        model_net=model_net,
        fps=10,
        image_keys=input_names+mask_names)

    # Dataset generates trajectories with shape [Bxslx...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    # Get train step
    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)
    train_step = common.function(train_step)

    if agent_name == 'latent_sac':
      def train_model_step():
        experience, _ = next(iterator)
        return tf_agent.train_model(experience)
      train_model_step = common.function(train_model_step)

    # Training initializations
    time_step = None
    time_acc = 0
    env_steps_before = env_steps.result().numpy()

    # Start training
    for iteration in range(num_iterations):
      start_time = time.time()

      if agent_name == 'latent_sac' and iteration < initial_model_train_steps:
        train_model_step()
      else:
        # Run collect
        time_step, _ = collect_driver.run(time_step=time_step)

        # Train an iteration
        for _ in range(train_steps_per_iteration):
          train_step()

      time_acc += time.time() - start_time

      # Log training information
      if global_step.numpy() % log_interval == 0:
        logging.info('env steps = %d, average return = %f', env_steps.result(),
                     average_return.result())
        env_steps_per_sec = (env_steps.result().numpy() -
                             env_steps_before) / time_acc
        logging.info('%.3f env steps/sec', env_steps_per_sec)
        tf.summary.scalar(
            name='env_steps_per_sec',
            data=env_steps_per_sec,
            step=env_steps.result())
        time_acc = 0
        env_steps_before = env_steps.result().numpy()

      # Get training metrics
      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=env_steps.result())

      # Evaluation
      if global_step.numpy() % eval_interval == 0:
        # Log evaluation metrics
        compute_summaries(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            train_step=global_step,
            summary_writer=summary_writer,
            num_episodes=num_eval_episodes,
            num_episodes_to_render=num_images_per_summary,
            model_net=model_net,
            fps=10,
            image_keys=input_names+mask_names)

      # Save checkpoints
      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)


def main():
    general_params, a2c_params, \
        pref_interface_params, rew_pred_training_params = parse_args()

    if general_params['debug']:
        logging.getLogger().setLevel(logging.DEBUG)

    run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params)


def run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params):
    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
    start_policy_training_flag = Queue(maxsize=1)

    if general_params['render_episodes']:
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = episode_renderer = None

    if a2c_params['env_id'] in ['MovingDot-v0', 'MovingDotNoFrameskip-v0']:
        reward_predictor_network = net_moving_dot_features
    elif a2c_params['env_id'] in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'CarRacing-v2','carla-v0']:
        reward_predictor_network = net_cnn
    else:
        raise Exception("Unsure about reward predictor network for {}".format(
            a2c_params['env_id']))

    def make_reward_predictor(name, cluster_dict):
        return RewardPredictorEnsemble(
            cluster_job_name=name,
            cluster_dict=cluster_dict,
            log_dir=general_params['log_dir'],
            batchnorm=rew_pred_training_params['batchnorm'],
            dropout=rew_pred_training_params['dropout'],
            lr=rew_pred_training_params['lr'],
            core_network=reward_predictor_network)

    save_make_reward_predictor(general_params['log_dir'],
                               make_reward_predictor)

    if general_params['mode'] == 'gather_initial_prefs':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)

        n_train = general_params['max_prefs'] * (1 - PREFS_VAL_FRACTION)
        n_val = general_params['max_prefs'] * PREFS_VAL_FRACTION
        pref_db_train = PrefDB(maxlen=n_train)
        pref_db_val = PrefDB(maxlen=n_val)
        pref_buffer = PrefBuffer(db_train=pref_db_train, db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        pref_buffer.wait_until_len(general_params['n_initial_prefs'])
        pref_db_train, pref_db_val = pref_buffer.get_dbs()

        save_prefs(general_params['log_dir'], pref_db_train, pref_db_val)

        pi_proc.terminate()
        pi.stop_renderer()
        a2c_proc.terminate()
        pref_buffer.stop_recv_thread()

        env.close()
    elif general_params['mode'] == 'pretrain_reward_predictor':
        cluster_dict = create_cluster_dict(['ps', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=True,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=None,
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'],
            log_dir=general_params['log_dir'])
        rpt_proc.join()
        ps_proc.terminate()
    elif general_params['mode'] == 'train_policy_with_original_rewards':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=False,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        start_policy_training_flag.put(True)
        a2c_proc.join()
        env.close()

    elif general_params['mode'] == 'train_policy_with_pretrained_predictor':

        cluster_dict = create_cluster_dict(['ps', 'a2c', 'train'])
        #print("!!!!!load param!!!!!!!!!!!!")
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        #print("!!!!!start policy training!!!!!!!!!!!!")
        env, a2c_proc = start_policy_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        a2c_proc.join()
        ps_proc.terminate()
        env.close()

    elif general_params['mode'] == 'train_policy_with_preferences':
        cluster_dict = create_cluster_dict(['ps', 'a2c', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=True,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=rew_pred_training_params['load_ckpt_dir'],
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'],
            log_dir=general_params['log_dir'])
        env, a2c_proc = start_policy_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        # We wait for A2C to complete the specified number of policy training
        # steps
        rpt_proc.join()
        a2c_proc.terminate()
        ps_proc.terminate()
        pi_proc.terminate()
        pi.stop_renderer()
        env.close()
    else:
        raise Exception("Unknown mode: {}".format(general_params['mode']))

    if episode_renderer:
        episode_renderer.stop()


def save_prefs(log_dir, pref_db_train, pref_db_val):
    train_path = osp.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    print("Saved training preferences to '{}'".format(train_path))
    val_path = osp.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    print("Saved validation preferences to '{}'".format(val_path))


def save_make_reward_predictor(log_dir, make_reward_predictor):
    save_dir = osp.join(log_dir, 'reward_predictor_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, 'make_reward_predictor.pkl'), 'wb') as fh:
        fh.write(cloudpickle.dumps(make_reward_predictor))


def create_cluster_dict(jobs):
    n_ports = len(jobs) + 1
    ports = get_port_range(start_port=2200,
                           n_ports=n_ports,
                           random_stagger=True)
    cluster_dict = {}
    for part, port in zip(jobs, ports):
        cluster_dict[part] = ['localhost:{}'.format(port)]
    return cluster_dict


def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])


def make_envs(env_id, n_envs, seed):
    def wrap_make_env(env_id, rank):
        def _thunk():
            return make_env(env_id, seed + rank)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv(env_id, [wrap_make_env(env_id, i)
                                 for i in range(n_envs)])
    return env




def start_parameter_server(cluster_dict, make_reward_predictor):
    def f():
        make_reward_predictor('ps', cluster_dict)
        while True:
            time.sleep(1.0)

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_policy_training(cluster_dict, make_reward_predictor, gen_segments,
                          start_policy_training_pipe, seg_pipe,
                          episode_vid_queue, log_dir, a2c_params):
    env_id = a2c_params['env_id']


    # Done here because daemonic processes can't have children
    env = make_envs(a2c_params['env_id'],
                    a2c_params['n_envs'],
                   a2c_params['seed'])
    
    mask_names = ['birdeye']
    input_names = ['camera', 'lidar']
    action_repeat = 4
    
    del a2c_params['env_id'], a2c_params['n_envs']

    ckpt_dir = osp.join(log_dir, 'policy_checkpoints')
    os.makedirs(ckpt_dir)
    print(a2c_params['gin_file'], a2c_params['gin_param'])

    def f(gen_segments, seg_pipe):
        if make_reward_predictor:
            reward_predictor = make_reward_predictor('a2c', cluster_dict)

        else:
            reward_predictor = None
            gen_segments = None
            seg_pipe = None
        py_env, eval_py_env = load_carla_env(env_name='carla-v0', obs_channels=input_names+mask_names, action_repeat=action_repeat,reward_predictor=reward_predictor,seg_pipe=seg_pipe,gen_segments=gen_segments)

        logging.set_verbosity(logging.INFO)
        gin.parse_config_file("params.gin") #a2c_params['gin_file'], a2c_params['gin_param'])

        train_eval("logs",
            "latent_sac",
            py_env=py_env, 
            eval_py_env=eval_py_env, 
            seg_pipe=seg_pipe,
            start_policy_training_pipe=start_policy_training_pipe,
            episode_vid_queue=episode_vid_queue,
            reward_predictor=reward_predictor,
            ckpt_save_dir=ckpt_dir,
            gen_segments=gen_segments)#a2c_params['root_dir'], a2c_params['experiment_name'])


    proc = Process(target=f, daemon=True, args=(gen_segments,seg_pipe))
    proc.start()
    return env, proc


def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs,
                         log_dir):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

    # Needs to be done in the main process because does GUI setup work
    prefs_log_dir = osp.join(log_dir, 'pref_interface')
    pi = PrefInterface(synthetic_prefs=synthetic_prefs,
                       max_segs=max_segs,
                       log_dir=prefs_log_dir)
    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc


def start_reward_predictor_training(cluster_dict,
                                    make_reward_predictor,
                                    just_pretrain,
                                    pref_pipe,
                                    start_policy_training_pipe,
                                    max_prefs,
                                    n_initial_prefs,
                                    n_initial_epochs,
                                    prefs_dir,
                                    load_ckpt_dir,
                                    val_interval,
                                    ckpt_interval,
                                    log_dir):
    def f():
        rew_pred = make_reward_predictor('train', cluster_dict)
        rew_pred.init_network(load_ckpt_dir)

        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!reward predictor training start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if prefs_dir is not None:
            train_path = osp.join(prefs_dir, 'train.pkl.gz')
            pref_db_train = PrefDB.load(train_path)
            print("Loaded training preferences from '{}'".format(train_path))
            n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))

            val_path = osp.join(prefs_dir, 'val.pkl.gz')
            pref_db_val = PrefDB.load(val_path)
            print("Loaded validation preferences from '{}'".format(val_path))
            n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))
        else:
            n_train = max_prefs * (1 - PREFS_VAL_FRACTION)
            n_val = max_prefs * PREFS_VAL_FRACTION
            pref_db_train = PrefDB(maxlen=n_train)
            pref_db_val = PrefDB(maxlen=n_val)

        pref_buffer = PrefBuffer(db_train=pref_db_train,
                                 db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        if prefs_dir is None:
            pref_buffer.wait_until_len(n_initial_prefs)

        save_prefs(log_dir, pref_db_train, pref_db_val)


        if not load_ckpt_dir:
            print("Pretraining reward predictor for {} epochs".format(
                n_initial_epochs))
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            for i in range(n_initial_epochs):
                # Note that we deliberately don't update the preferences
                # databases during pretraining to keep the number of
                # fairly preferences small so that pretraining doesn't take too
                # long.
                print("Reward predictor training epoch {}".format(i))

                #print("!!!!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                rew_pred.train(pref_db_train, pref_db_val, val_interval)

                #print("!!!!!!!!!!!!!!!!!!!!!!test2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if i and i % ckpt_interval == 0:
                    rew_pred.save()
            print("Reward predictor pretraining done")
            rew_pred.save()

        if just_pretrain:
            return

        start_policy_training_pipe.put(True)
        
        i = 0
        while True:
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            save_prefs(log_dir, pref_db_train, pref_db_val)
            #print("!!!!!!!!!!!!rew_pred.train!!!!!!!!!!!!")
            rew_pred.train(pref_db_train, pref_db_val, val_interval)
            #print("rew_pred.train end")
            if i and i % ckpt_interval == 0:
                rew_pred.save()

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_episode_renderer():
    episode_vid_queue = Queue()
    renderer = VideoRenderer(
        episode_vid_queue,
        playback_speed=8,
        zoom=1,
        mode=VideoRenderer.play_through_mode)
    return episode_vid_queue, renderer


if __name__ == '__main__':
    main()
