from config_meta import MetaConfig
from absl import flags
from absl import app
import tensorflow as tf
import core
import distributions
from tictactoe import env
from tictactoe import network
from learner import Learner

OPTIMIZER                           = flags.DEFINE_string('optimizer',
                                                        'adam',
                                                        'One of [sgd, adam, rmsprop, adagrad]')
LEARNING_RATE                       = flags.DEFINE_float('learning_rate',
                                                        1e-3,
                                                        'Learning rate.')
MOMENTUM                            = flags.DEFINE_float('momentum',
                                                        0.9,
                                                        'Momentum')
LR_DECAY_FRACTION                   = flags.DEFINE_float('lr_decay_fraction',
                                                        0.01,
                                                        'Final LR as a fraction of initial.')
LR_WARM_RESTART                     = flags.DEFINE_integer('lr_warm_restarts',
                                                        1,
                                                        'Do warm restarts for LR decay.')
LR_DECAY_STEP                       = flags.DEFINE_integer('lr_decay_steps',
                                                        int(2e4),
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

flags.DEFINE_integer('value_encoder_steps', 8, 'If 0, take 1 step per integer')
flags.DEFINE_integer('reward_encoder_steps', None,
                     'If None, take over the value from value_encoder_steps')

FLAGS = flags.FLAGS

def create_agent(env_descriptor, parametric_action_distribution):
  reward_encoder_steps = FLAGS.reward_encoder_steps
  if reward_encoder_steps is None:
    reward_encoder_steps = FLAGS.value_encoder_steps

  reward_encoder = core.ValueEncoder(
      *env_descriptor.reward_range,
      reward_encoder_steps,
      use_contractive_mapping=False)
  value_encoder = core.ValueEncoder(
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

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  env_descriptor = env.get_descriptor()
  parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(env_descriptor.action_space)
  agent = create_agent(env_descriptor, parametric_action_distribution)

  # Known bounds for Q-values have to include rewards and values.
  known_bounds = core.KnownBounds(
      *map(sum, zip(env_descriptor.reward_range, env_descriptor.value_range)))

  config = MetaConfig(learner_from_flag=True, actor_from_flag=True, muzero_from_flag=True)
  config.build_meta_config(env_descriptor.action_space.n, known_bounds=known_bounds)
  config.make_specs(env_descriptor, agent)
  L = Learner(config , agent, create_optimizer, env_descriptor)
  L.train()


if __name__ == '__main__':
  app.run(main)