from time import sleep
import tensorflow as tf
from learner_config import LearnerConfig
import BufferUtils as utils
import core as mzcore
#from tictactoe.muzero_main import 
from tictactoe import env
from tictactoe import network
from absl import flags
from absl import logging
from absl import app
import distributions
import learner_flagsv2 as learner_flags
import numpy as np
import pickle
#import random
import pathlib
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
flags.DEFINE_string('episode_dir', 'actor_0', 'The path to the directory that stores episodes.')

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

def make_spec_from_gym_space(space, name):
  if space.dtype is not None:
    specs = tf.TensorSpec(space.shape, space.dtype, name)
  else:
    # This is a tuple space
    specs = tuple(
        make_spec_from_gym_space(s, '{}_{}'.format(name, idx))
        for idx, s in enumerate(space.spaces))
  return specs

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
      episode_dir="actor_0",
      pb_c_base=FLAGS.pb_c_base,
      pb_c_init=FLAGS.pb_c_init,
      known_bounds=known_bounds,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      use_softmax_for_action_selection=(
          FLAGS.use_softmax_for_action_selection == 1),
      max_num_action_expansion=FLAGS.max_num_action_expansion)
  
  config = LearnerConfig
  num_target_steps = mzconfig.num_unroll_steps + 1
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
  replay_buffer_specs = (
      observation_specs,
      tf.TensorSpec(
          env_descriptor.action_space.shape + (mzconfig.num_unroll_steps,),
          env_descriptor.action_space.dtype, 'history'),
      *target_specs,
  )

  batch_specs = (
      tf.TensorSpec([LearnerConfig.batch_size], tf.float32, 'priorities'),
      tf.TensorSpec([LearnerConfig.batch_size] + list(env_descriptor.observation_space.shape), tf.float32, 'observation'),
      tf.TensorSpec([LearnerConfig.batch_size] + list(env_descriptor.action_space.shape + (mzconfig.num_unroll_steps,)), tf.int64, 'history'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps], tf.float32, 'value_mask'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps], tf.float32, 'reward_mask'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps], tf.float32, 'policy_mask'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps], tf.float32, 'value'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps], tf.float32, 'reward'),
      tf.TensorSpec([LearnerConfig.batch_size, num_target_steps, env_descriptor.action_space.n], tf.float32, 'policy'))

  class DataLoader:
    def __init__(self, replay_buffer_specs, lrn_cfg, mz_cfg):
      # Load
      self.mz_cfg = mz_cfg
      self.replay_buffer_specs = replay_buffer_specs
      self.lrn_cfg = lrn_cfg
      # Create replay buffer
      self.replay_buffer = utils.PrioritizedReplay(
          self.lrn_cfg.replay_buffer_size,
          self.replay_buffer_specs,
          self.lrn_cfg.importance_sampling_exponent,
      )
      self.cache = []

    def sample(self):
      indices, weights, replays = self.replay_buffer.sample(self.lrn_cfg.batch_size, self.lrn_cfg.priority_sampling_exponent)
      if self.lrn_cfg.replay_buffer_update_priority_after_sampling_value >= 0.:
        self.replay_buffer.update_priorities(
            indices,
            tf.convert_to_tensor(
                np.ones(indices.shape) *
                self.lrn_cfg.replay_buffer_update_priority_after_sampling_value,
                dtype=tf.float32))
      data = (weights, *replays)
      return tf.nest.flatten(data)
    
    def loadBatchesGen(self):
      directory = pathlib.Path(self.mz_cfg.episode_dir).expanduser()
      print(directory)
      cache = []
      while tf.constant(True):
        for filename in directory.glob('*.pkl'):
          if filename not in cache:
            try:
              with filename.open('rb') as f:
                batch = pickle.load(f)
            except Exception as e:
              print(f'Could not load episode: {e}')
              continue
            cache.append(filename)
            tf.print(len(cache))
            tf.print(self.replay_buffer.num_inserted)
            priorities, *samples = batch
            self.replay_buffer.insert(tuple(samples), priorities)
            if self.replay_buffer.num_inserted <= self.lrn_cfg.replay_buffer_size:
              tf.print('waiting for replay buffer to fill. Status:', self.replay_buffer.num_inserted, ' / ', self.lrn_cfg.replay_buffer_size)
            else:
              tf.print('Replay buffer filled with ', self.replay_buffer.num_inserted,' samples')
        yield tuple(self.sample())
    
    def makeGenerator(self):
        print(batch_specs)
        return tf.data.Dataset.from_generator(self.loadBatchesGen, output_signature=tuple(batch_specs))
 
  DL = DataLoader(replay_buffer_specs, config, mzconfig)
  dataset = DL.makeGenerator()#tf.data.Dataset.from_generator(DL.loadBatchesGen, output_signature=tuple(batch_specs))
  it = iter(dataset)
  next(it)

if __name__ == '__main__':
  app.run(main)