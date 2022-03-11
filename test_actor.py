from absl import app
from absl import flags
import tensorflow as tf

import config_meta
import core as mzcore
import distributions
from tictactoe import env
from tictactoe import network

from agent import Agent

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

  def get_add_batch_size(batch_size):
    def add_batch_size(ts):
      return tf.TensorSpec([batch_size] + list(ts.shape), ts.dtype, ts.name)
    return add_batch_size

  @tf.function(
        input_signature=tf.nest.map_structure(
            get_add_batch_size(config.lrn.initial_inference_batch_size),
            config.specs.initial_inference))
  def initial_inference(observation):
    return agent.initial_inference(observation)

  @tf.function(
      input_signature=tf.nest.map_structure(
          get_add_batch_size(config.lrn.recurrent_inference_batch_size),
          config.specs.recurrent_inference))
  def recurrent_inference(hidden_state, action):
    return agent.recurrent_inference(hidden_state, action)

  A = Agent(env.create_environment, config)

  logdir = "/home/antoine/ws/muzero/test/learner"
  ckpt = tf.train.Checkpoint(agent=agent)
  manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=10, keep_checkpoint_every_n_hours=6)
  try:
    ckpt.restore(manager.latest_checkpoint).assert_nontrivial_match()
  except:
    pass
  for i in range(2000):
    A.playEpisode(initial_inference, recurrent_inference, 0)
    try:
      ckpt.restore(manager.latest_checkpoint).assert_nontrivial_match()
    except:
      pass

if __name__ == '__main__':
  app.run(main)
