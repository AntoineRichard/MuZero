from time import sleep
import tensorflow as tf
import BufferUtils as utils
import core as mzcore
#from tictactoe.muzero_main import 
from tictactoe import env
from tictactoe import network
from absl import flags
from absl import logging
from absl import app
import distributions
import numpy as np
import pickle
import pathlib
import config_meta

from dataloader import DataLoader

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

flags.DEFINE_integer('value_encoder_steps', 8, 'If 0, take 1 step per integer')
flags.DEFINE_integer('reward_encoder_steps', None,
                     'If None, take over the value from value_encoder_steps')

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

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  env_descriptor = env.get_descriptor()
  parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(env_descriptor.action_space)
  agent = create_agent(env_descriptor, parametric_action_distribution)

  # Known bounds for Q-values have to include rewards and values.
  known_bounds = mzcore.KnownBounds(
      *map(sum, zip(env_descriptor.reward_range, env_descriptor.value_range)))

  config = config_meta.MetaConfig(learner_from_flag=True, actor_from_flag=True, muzero_from_flag=True)
  config.build_meta_config(env_descriptor.action_space.n, known_bounds=known_bounds)
  config.make_specs(env_descriptor, agent)
  print(config.specs.replay_buffer)
  
 
  DL = DataLoader(config)
  dataset = DL.makeGenerator()#tf.data.Dataset.from_generator(DL.loadBatchesGen, output_signature=tuple(batch_specs))
  it = iter(dataset)
  next(it)
  next(it)

if __name__ == '__main__':
  app.run(main)