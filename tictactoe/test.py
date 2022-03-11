# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
"""MuZero."""

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np
import random
import collections

import sys

sys.path.append("..")

#import actor
from learner_config import LearnerConfig
import core as mzcore
import distributions
import BufferUtils as utils
#import learner
import learner_flagsv2 as learner_flags
from tictactoe import env
from tictactoe import network

USE_SOFTMAX_FOR_TARGET = flags.DEFINE_integer(
    'use_softmax_for_target', 0,
    'If True (1), use a softmax for the child_visit count distribution that '
    'is used as a target for the policy.')
NUM_TEST_ACTORS = flags.DEFINE_integer(
    'num_test_actors', 2, 'Number of actors that are used for testing.')
NUM_ACTORS_WITH_SUMMARIES = flags.DEFINE_integer(
    'num_actors_with_summaries', 1,
    'Number of actors that will log debug/profiling TF '
    'summaries.')
ACTOR_LOG_FREQUENCY = flags.DEFINE_integer('actor_log_frequency', 10,
                                           'in number of training steps')
MCTS_VIS_FILE = flags.DEFINE_string(
    'mcts_vis_file', None, 'File in which to log the mcts visualizations.')
FLAG_FILE = flags.DEFINE_string('flag_file', None,
                                'File in which to log the parameters.')
ENABLE_ACTOR_LOGGING = flags.DEFINE_boolean('enable_actor_logging', True,
                                            'Verbose logging for the actor.')
MAX_NUM_ACTION_EXPANSION = flags.DEFINE_integer(
    'max_num_action_expansion', 0,
    'Maximum number of new nodes for a node expansion. 0 for no limit. '
    'This is important for the full vocabulary.')
ACTOR_ENQUEUE_EVERY = flags.DEFINE_integer(
    'actor_enqueue_every', 0,
    'After how many steps the actor enqueues samples. 0 for at episode end.')
ACTOR_SKIP = flags.DEFINE_integer('actor_skip', 0,
                                  'How many target samples the actor skips.')

flags.DEFINE_string('optimizer', 'adam', 'One of [sgd, adam, rmsprop, adagrad]')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_float('lr_decay_fraction', 0.01,
                   'Final LR as a fraction of initial.')
flags.DEFINE_integer('lr_warm_restarts', 1, 'Do warm restarts for LR decay.')
flags.DEFINE_integer('lr_decay_steps', int(2e4),
                     'Decay steps for the cosine learning rate schedule.')
flags.DEFINE_integer('n_mlp_layers', 2, 'Number of MLP hidden layers.')
flags.DEFINE_integer('mlp_size', 512, 'Sizes of each of MLP hidden layer.')
flags.DEFINE_integer(
    'n_lstm_layers', 2,
    'Number of LSTM layers. LSTM layers afre applied after MLP layers.')
flags.DEFINE_integer('lstm_size', 128, 'Sizes of each LSTM layer.')
flags.DEFINE_integer('n_head_hidden_layers', 2,
                     'Number of hidden layers in heads.')
flags.DEFINE_integer('head_hidden_size', 512,
                     'Sizes of each head hidden layer.')

flags.DEFINE_integer('num_simulations', 64, 'Number of simulations.')
flags.DEFINE_integer('td_steps', -1, 'Number of TD steps.')
flags.DEFINE_integer('num_unroll_steps', 5, 'Number of unroll steps.')
flags.DEFINE_float('one_minus_discount', .003, 'One minus discount factor.')
flags.DEFINE_float('dirichlet_alpha', .5, 'Dirichlet alpha.')
flags.DEFINE_float('root_exploration_fraction', .25,
                   'Root exploration fraction.')
flags.DEFINE_integer('pb_c_base', 19652, 'PB C Base.')
flags.DEFINE_float('pb_c_init', 2.5, 'PB C Init.')

flags.DEFINE_float('temperature', .1, 'for softmax sampling of actions')

flags.DEFINE_integer('value_encoder_steps', 8, 'If 0, take 1 step per integer')
flags.DEFINE_integer('reward_encoder_steps', None,
                     'If None, take over the value from value_encoder_steps')
flags.DEFINE_integer(
    'play_max_after_moves', -1,
    'Play the argmax after this many game moves. -1 means never play argmax')
flags.DEFINE_integer(
    'use_softmax_for_action_selection', 0,
    'Whether to use softmax (1) for regular histogram sampling (0).')

FLAGS = flags.FLAGS


def create_agent(env_descriptor, parametric_action_distribution):
  reward_encoder_steps = FLAGS.reward_encoder_steps
  if reward_encoder_steps is None:
    reward_encoder_steps = FLAGS.value_encoder_steps

  reward_encoder = mzcore.ValueEncoder(
      *env_descriptor.reward_range,
      reward_encoder_steps,
      use_contractive_mapping=False)
  value_encoder = mzcore.ValueEncoder(
      *env_descriptor.value_range,
      FLAGS.value_encoder_steps,
      use_contractive_mapping=False)
  return network.MLPandLSTM(
      mlp_sizes=[FLAGS.mlp_size] * FLAGS.n_mlp_layers,
      parametric_action_distribution=parametric_action_distribution,
      rnn_sizes=[FLAGS.lstm_size] * FLAGS.n_lstm_layers,
      head_hidden_sizes=[FLAGS.head_hidden_size] * FLAGS.n_head_hidden_layers,
      reward_encoder=reward_encoder,
      value_encoder=value_encoder)


def create_optimizer(unused_final_iteration):
  if FLAGS.lr_warm_restarts:
    learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
        FLAGS.learning_rate,
        FLAGS.lr_decay_steps,
        alpha=FLAGS.lr_decay_fraction)
  else:
    learning_rate_fn = tf.keras.experimental.CosineDecay(
        FLAGS.learning_rate,
        FLAGS.lr_decay_steps,
        alpha=FLAGS.lr_decay_fraction)
  if FLAGS.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate_fn, momentum=FLAGS.momentum)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.keras.optimizers.AdaGrad(learning_rate_fn)
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate_fn, momentum=FLAGS.momentum)
  else:
    raise ValueError('Unknown optimizer: {}'.format(FLAGS.optimizer))
  return optimizer, learning_rate_fn

def _create_training_samples(batch_queue, mzconfig, episode, start_idx=0):
  start_idx += random.choice(range(ACTOR_SKIP.value + 1))
  for i in range(start_idx, len(episode.history), ACTOR_SKIP.value + 1):
    target = episode.make_target(
        state_index=i,
        num_unroll_steps=mzconfig.num_unroll_steps,
        td_steps=mzconfig.td_steps,
        rewards=episode.rewards,
        policy_distributions=episode.child_visits,
        discount=episode.discount,
        value_approximations=episode.root_values)
    #print(target)
    priority = np.float32(1e-2)  # preventing all zero priorities
    if len(episode) > 0:  # pylint: disable=g-explicit-length-test
      last_value_idx = min(len(episode) - 1 - i, len(target.value) - 1)
      priority = np.maximum(
          priority,
          np.float32(
              np.abs(episode.root_values[i + last_value_idx] -
                     target.value[last_value_idx])))

    # This will be batched and given to add_to_replay_buffer on the
    # learner.
    priority = tf.squeeze(priority)
    sample = (
        priority,
        episode.make_image(i),
        tf.stack(
            episode.history_range(i, i + mzconfig.num_unroll_steps)),
    ) + tuple(map(lambda x: tf.cast(tf.stack(x), tf.float32), target))
    batch_queue.append(sample)
    print(sample)
  if ENABLE_ACTOR_LOGGING.value:
    logging.info(
        'Added %d samples to the batch_queue. Size: %d of needed %d',
        len(episode.history) - start_idx, len(batch_queue),
        mzconfig.train_batch_size)



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  def visit_softmax_temperature(num_moves, training_steps, is_training=True):
    if not is_training:
      return 0.
    if FLAGS.play_max_after_moves < 0:
      return FLAGS.temperature
    if num_moves < FLAGS.play_max_after_moves:
      return FLAGS.temperature
    else:
      return 0.

  env_descriptor = env.get_descriptor()

  # Known bounds for Q-values have to include rewards and values.
  known_bounds = mzcore.KnownBounds(
      *map(sum, zip(env_descriptor.reward_range, env_descriptor.value_range)))
  mzconfig = mzcore.MuZeroConfig(
      action_space_size=env_descriptor.action_space.n,
      max_moves=env_descriptor.action_space.n,
      discount=1.0 - FLAGS.one_minus_discount,
      dirichlet_alpha=FLAGS.dirichlet_alpha,
      root_exploration_fraction=FLAGS.root_exploration_fraction,
      num_simulations=FLAGS.num_simulations,
      initial_inference_batch_size=1,#(
          #learner_flags.INITIAL_INFERENCE_BATCH_SIZE.value),
      recurrent_inference_batch_size=1,#(
          #learner_flags.RECURRENT_INFERENCE_BATCH_SIZE.value),
      #initial_inference_batch_size=(
      #    learner_flags.INITIAL_INFERENCE_BATCH_SIZE.value),
      #recurrent_inference_batch_size=(
      #    learner_flags.RECURRENT_INFERENCE_BATCH_SIZE.value),
      train_batch_size=learner_flags.BATCH_SIZE.value,
      td_steps=FLAGS.td_steps,
      num_unroll_steps=FLAGS.num_unroll_steps,
      pb_c_base=FLAGS.pb_c_base,
      pb_c_init=FLAGS.pb_c_init,
      known_bounds=known_bounds,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      use_softmax_for_action_selection=(
          FLAGS.use_softmax_for_action_selection == 1),
      max_num_action_expansion=FLAGS.max_num_action_expansion)

  training_steps = 0
  actor_step = tf.Variable(0, dtype=tf.int64)
  parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(env_descriptor.action_space)
  agent = create_agent(env_descriptor, parametric_action_distribution)
  optimizer, learning_rate_fn = create_optimizer(LearnerConfig.total_iterations)

  def make_spec_from_gym_space(space, name):
    if space.dtype is not None:
      specs = tf.TensorSpec(space.shape, space.dtype, name)
    else:
      # This is a tuple space
      specs = tuple(
          make_spec_from_gym_space(s, '{}_{}'.format(name, idx))
          for idx, s in enumerate(space.spaces))
    return specs

  def get_add_batch_size(batch_size):

    def add_batch_size(ts):
      return tf.TensorSpec([batch_size] + list(ts.shape), ts.dtype, ts.name)

    return add_batch_size

  num_target_steps = mzconfig.num_unroll_steps + 1
  replay_buffer_size = LearnerConfig.replay_buffer_size
  initial_agent_state = agent.initial_state(1)
  target_specs = (
      tf.TensorSpec([num_target_steps], tf.float32, 'value_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'reward_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'policy_mask'),
      tf.TensorSpec([num_target_steps], tf.float32, 'value'),
      tf.TensorSpec([num_target_steps], tf.float32, 'reward'),
      tf.TensorSpec([num_target_steps, env_descriptor.action_space.n],
                    tf.float32, 'policy'),
  )
  observation_specs = make_spec_from_gym_space(env_descriptor.observation_space, 'observation')
  replay_buffer_specs = (observation_specs,
    tf.TensorSpec(env_descriptor.action_space.shape + (mzconfig.num_unroll_steps,),
                 env_descriptor.action_space.dtype, 'history'), *target_specs)
  replay_buffer = utils.PrioritizedReplay(replay_buffer_size, replay_buffer_specs, LearnerConfig.importance_sampling_exponent)
  replay_queue_specs = (tf.TensorSpec([], tf.float32, 'priority'), *replay_buffer_specs)
  action_specs = make_spec_from_gym_space(env_descriptor.action_space, 'action')
  agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  initial_inference_specs = (observation_specs,)
  recurrent_inference_specs = (agent_state_specs, action_specs)

  replay_queue_size = LearnerConfig.replay_queue_size
  replay_buffer_queue = utils.StructuredFIFOQueue(replay_queue_size,
                                                  replay_queue_specs)

  @tf.function(
        input_signature=tf.nest.map_structure(
            get_add_batch_size(mzconfig.initial_inference_batch_size),
            initial_inference_specs))
  def initial_inference(observation):
    return agent.initial_inference(observation)

  @tf.function(
      input_signature=tf.nest.map_structure(
          get_add_batch_size(mzconfig.recurrent_inference_batch_size),
          recurrent_inference_specs))
  def recurrent_inference(hidden_state, action):
    return agent.recurrent_inference(hidden_state, action)
  
  @tf.function(input_signature=[])
  def learning_iteration():
    return optimizer.iterations

  @tf.function(
      input_signature=tf.nest.map_structure(
          get_add_batch_size(LearnerConfig.batch_size), replay_queue_specs))
  def add_to_replay_buffer(*batch):
    print('We re in !')
    queue_size = replay_buffer_queue.size()
    num_free = replay_queue_size - queue_size
    print('It s doing stuff')
    if not LearnerConfig.replay_queue_block and num_free < LearnerConfig.recurrent_inference_batch_size:
      replay_buffer_queue.dequeue_many(LearnerConfig.recurrent_inference_batch_size)
    print('It s OK')
    replay_buffer_queue.enqueue_many(batch)

  def _add_queue_to_replay_buffer(batch_queue, mzconfig):
    print("In add Queue to RB")
    while len(batch_queue) >= mzconfig.train_batch_size:
      print("In While loop of AQ2RB")
      batch = [
          batch_queue.popleft()
          for _ in range(mzconfig.train_batch_size)
      ]
      print("Batch built")
      flat_batch = [tf.nest.flatten(b) for b in batch]
      stacked_batch = list(map(tf.stack, zip(*flat_batch)))
      structured_batch = tf.nest.pack_sequence_as(
          batch[0], stacked_batch)
      print("Batch structured")
      #print(structured_batch)
      add_to_replay_buffer(*structured_batch)
      print("batch_added")
      if ENABLE_ACTOR_LOGGING.value:
        logging.info('Added batch of size %d into replay_buffer.',
                     len(batch))
      print(replay_buffer_queue)

  is_training_actor = True

  batch_queue = collections.deque()

  for i in range(100):
    episode = mzconfig.new_episode(env.create_environment(0,False))
    legal_actions_fn = episode.legal_actions
    last_enqueued_idx = 0
    while (not episode.terminal() and len(episode.history) < mzconfig.max_moves):
      # This loop is the agent playing the episode.
      current_observation = episode.make_image(-1)
      current_observation = np.expand_dims(current_observation,0)
      #print(current_observation.shape)

      # Map the observation to hidden space.
      initial_inference_output = initial_inference(current_observation)
      initial_inference_output = tf.nest.map_structure(lambda t: t.numpy(), initial_inference_output)
      #initial_inference_output = tf.nest.map_structure(lambda t: np.squeeze(t.numpy()), initial_inference_output)
      root = mzcore.prepare_root_node(mzconfig, legal_actions_fn(), initial_inference_output)
      mzcore.run_mcts(mzconfig, root, episode.action_history(), legal_actions_fn,
       recurrent_inference, episode.visualize_mcts)
      action = mzcore.select_action(mzconfig, len(episode.history), root, train_step=actor_step.numpy(),
       use_softmax=mzconfig.use_softmax_for_action_selection, is_training=False)

      try:
        episode.apply(action=action, training_steps=training_steps)
        training_steps = learning_iteration().numpy()
      except mzcore.RLEnvironmentError as env_error:
        logging.warning('Environment failed: %s', str(env_error))
        episode.failed = True
        break
      
      episode.store_search_statistics(root, use_softmax=(USE_SOFTMAX_FOR_TARGET.value == 1))
      actor_step.assign_add(delta=1)
      if is_training_actor and ACTOR_ENQUEUE_EVERY.value > 0 and (len(episode.history) - last_enqueued_idx) >= ACTOR_ENQUEUE_EVERY.value:
        _create_training_samples(batch_queue, mzconfig, episode, start_idx=last_enqueued_idx)
        last_enqueued_idx = len(episode.history)
        _add_queue_to_replay_buffer()

      print(action)
      #print(training_steps)
    _create_training_samples(batch_queue, mzconfig, episode, start_idx=last_enqueued_idx)
    #print([i.shape for i in batch_queue[-1]])
    #print([i.shape for i in batch_queue[-1]])
    _add_queue_to_replay_buffer(batch_queue, mzconfig)


if __name__ == '__main__':
  app.run(main)
