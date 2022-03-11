import concurrent.futures
from settings_learner import LearnerConfig
from config_meta import MetaConfig
from absl import logging
from absl import flags
from absl import app
import tensorflow as tf
import collections
import numpy as np
import copy
import time
import core
import os
import distributions
from losses import compute_loss, compute_pretrain_loss
import BufferUtils as utils
from dataloader import DataLoader
from tictactoe import env
from tictactoe import network


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

#def learner_loop(env_descriptor,
#                 agent,
#                 create_optimizer_fn,
#                 config: MetaConfig,
#                 pretraining=False):
#  """Main learner loop.
#
#  Args:
#    env_descriptor: An instance of utils.EnvironmentDescriptor.
#    create_agent_fn: Function that must create a new tf.Module with the neural
#      network that outputs actions and new agent state given the environment
#      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
#      example. The factory function takes as input the environment descriptor
#      and a parametric distribution over actions.
#    create_optimizer_fn: Function that takes the final iteration as argument and
#      must return a tf.keras.optimizers.Optimizer and a
#      tf.keras.optimizers.schedules.LearningRateSchedule.
#    config: A LearnerConfig object.
#    mzconfig: A MuZeroConfig object.
#    pretraining: Do pretraining.
#  """
#  logging.info('Starting learner loop')
#  settings = utils.init_learner(config.lrn.num_training_tpus)
#  strategy, inference_devices, training_strategy, encode, decode = settings
#  tf_function = noop_decorator if config.lrn.debug else tf.function
#
#  parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(
#      env_descriptor.action_space)
#
#  #if pretraining:
#  #  assert env_descriptor.pretraining_space is not None, (
#  #      'Must define a pretraining space')
#  #  pretraining_specs = make_spec_from_gym_space(
#  #      env_descriptor.pretraining_space, 'pretraining')
#
#  # Initialize agent and variables.
#  #with strategy.scope():
#  #agent = create_agent_fn(env_descriptor, parametric_action_distribution)
#  initial_agent_state = agent.initial_state(1)
#  if config.lrn.debug:
#    logging.info('initial state:\n{}'.format(initial_agent_state))
#
#  zero_observation = tf.nest.map_structure(
#      lambda s: tf.zeros([1] + list(s.shape), s.dtype), config.specs.observation)
#  zero_action = tf.nest.map_structure(
#      lambda s: tf.zeros([1] + list(s.shape), s.dtype), config.specs.action)
#
#  zero_initial_args = [encode(zero_observation)]
#  zero_recurrent_args = [encode(initial_agent_state), encode(zero_action)]
#  if config.lrn.debug:
#    logging.info('zero initial args:\n{}'.format(zero_initial_args))
#    logging.info('zero recurrent args:\n{}'.format(zero_recurrent_args))
#
#  #if pretraining:
#  #  zero_pretraining = tf.nest.map_structure(
#  #      lambda s: tf.zeros([1] + list(s.shape), s.dtype), pretraining_specs)
#  #  zero_pretraining_args = [encode(zero_pretraining)]
#  #  logging.info('zero pretraining args:\n{}'.format(zero_pretraining_args))
#  #else:
#  #zero_pretraining_args = None
#
#  #with strategy.scope():
#  #def create_variables(initial_args, recurrent_args, pretraining_args):
#  #  agent.initial_inference(*map(decode, initial_args))
#  #  agent.recurrent_inference(*map(decode, recurrent_args))
#  #  if pretraining_args is not None:
#  #    agent.pretraining_loss(*map(decode, pretraining_args))
#
#  # This complicates BatchNormalization, can't use it.
#  #create_variables(zero_initial_args, zero_recurrent_args,
#  #                 zero_pretraining_args)
#
#  #with strategy.scope():
#  # Create optimizer.
#  optimizer, learning_rate_fn = create_optimizer_fn(config.lrn.total_iterations)
#
#  # pylint: disable=protected-access
#  iterations = optimizer.iterations
#  optimizer._create_hypers()
#  optimizer._create_slots(
#      agent.get_trainable_variables(pretraining=pretraining))
#  # pylint: enable=protected-access
#
#  #with strategy.scope():
#  # ON_READ causes the replicated variable to act as independent variables for
#  # each replica.
#  #temp_grads = [
#  #    tf.Variable(
#  #        tf.zeros_like(v),
#  #        trainable=False,
#  #        synchronization=tf.VariableSynchronization.ON_READ,
#  #        name='temp_grad_{}'.format(v.name),
#  #    ) for v in agent.get_trainable_variables(pretraining=pretraining)
#  #]
#
#  logging.info('--------------------------')
#  logging.info('TRAINABLE VARIABLES')
#  for v in agent.get_trainable_variables(pretraining=pretraining):
#    logging.info('{}: {} | {}'.format(v.name, v.shape, v.dtype))
#  logging.info('--------------------------')
#
#  @tf_function
#  def _compute_loss(*args, **kwargs):
#    if pretraining:
#      return compute_pretrain_loss(config.lrn, *args, **kwargs)
#    else:
#      return compute_loss(config.lrn, *args, **kwargs)
#
#  @tf_function
#  def minimize(iterator):
#    data = next(iterator)
#
#    args = tf.nest.pack_sequence_as(config.specs.weighted_replay_buffer,decode(data, data))
#    with tf.GradientTape() as tape:
#      loss, logs = _compute_loss(parametric_action_distribution, agent, *args)
#    grads = tape.gradient(loss, agent.get_trainable_variables(pretraining=pretraining))
#    grads = grads
#    if config.lrn.gradient_norm_clip > 0.:
#      grads, _ = tf.clip_by_global_norm(grads, config.lrn.gradient_norm_clip)
#    optimizer.apply_gradients(
#        zip(grads, agent.get_trainable_variables(pretraining=pretraining)))
#    return logs
#
#  ## Logging.
#  logdir = os.path.join(config.lrn.logdir, 'learner')
#  summary_writer = tf.summary.create_file_writer(
#      logdir,
#      flush_millis=config.lrn.flush_learner_log_every_n_s * 1000,
#      max_queue=int(1E6))
#
#
#  #  if pretraining:
#  #    replay_buffer_specs = pretraining_specs
#  #  else:
#  #    replay_buffer_specs = (
#  #        observation_specs,
#  #        tf.TensorSpec(
#  #            env_descriptor.action_space.shape + (mzconfig.num_unroll_steps,),
#  #            env_descriptor.action_space.dtype, 'history'),
#  #        *target_specs,
#  #    )
#
#  #  weighted_replay_buffer_specs = (
#  #      tf.TensorSpec([], tf.float32, 'importance_weights'), *replay_buffer_specs)
#
#  #  episode_stat_specs = (
#  #      tf.TensorSpec([], tf.string, 'summary_name'),
#  #      tf.TensorSpec([], tf.float32, 'reward'),
#  #      tf.TensorSpec([], tf.int64, 'episode_length'),
#  #  )
#  #  if env_descriptor.extras:
#  #    episode_stat_specs += tuple(
#  #        tf.TensorSpec([], stat[1], stat[0])
#  #        for stat in env_descriptor.extras.get('learner_stats', []))
#
#  #  episode_stat_queue = utils.StructuredFIFOQueue(-1, episode_stat_specs)
#
#  #@tf.function(input_signature=episode_stat_specs)
#  #def add_to_reward_queue(*stats):
#  #  episode_stat_queue.enqueue(stats)
#
#  # Execute learning and track performance.
#  #with summary_writer.as_default(), \
#  #     concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#  DL = DataLoader(config)
#  dataset = DL.makeGenerator()#tf.data.Dataset.from_generator(DL.loadBatchesGen, output_signature=tuple(batch_specs))
#  it = iter(dataset)
#  print("GOING IN FOR THE TRAIN")
#  with summary_writer.as_default():#, \
#       #concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#    #log_future = executor.submit(lambda: None)  # No-op future.
#    #last_iterations = iterations
#    #last_log_time = time.time()
#    #values_to_log = collections.defaultdict(lambda: [])
#    while iterations < config.lrn.total_iterations:
#      #tf.summary.experimental.set_step(iterations)
#      print(iterations)
#      # Save checkpoint.
#      #current_time = time.time()
#      #if current_time - last_ckpt_time >= config.save_checkpoint_secs:
#      #  manager.save()
#      #  if config.export_agent:
#      #    # We also export the agent as a SavedModel to be used for inference.
#      #    saved_model_dir = os.path.join(logdir, 'saved_model')
#      #    network.export_agent_for_initial_inference(
#      #        agent=agent,
#      #        model_dir=os.path.join(saved_model_dir, 'initial_inference'))
#      #    network.export_agent_for_recurrent_inference(
#      #        agent=agent,
#      #        model_dir=os.path.join(saved_model_dir, 'recurrent_inference'))
#      #  last_ckpt_time = current_time
#
#      ##def log(iterations):
#      #  """Logs batch and episodes summaries."""
#      #  nonlocal last_iterations, last_log_time
#      #  #summary_writer.set_as_default()
#      #  #tf.summary.experimental.set_step(iterations)
#
#      #  # log data from the current minibatch
#      #  #for key, values in copy.deepcopy(values_to_log).items():
#      #  #  if values:
#      #  #    tf.summary.scalar(key, values[-1])  # could also take mean
#      #  #values_to_log.clear()
#      #  #tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
#      #  #tf.summary.scalar('replay_queue_size', replay_buffer_queue.size())
#      #  #stats = episode_stat_queue.dequeue_many(episode_stat_queue.size())
#
#      #  #summary_name_idx = [spec.name for spec in episode_stat_specs
#      #  #                   ].index('summary_name')
#      #  #summary_name_stats = stats[summary_name_idx]
#      #  #unique_summary_names, unique_summary_name_idx = tf.unique(
#      #  #    summary_name_stats)
#
#      #  #def log_mean_value(values, label):
#      #  #  mean_value = tf.reduce_mean(tf.cast(values, tf.float32))
#      #  #  tf.summary.scalar(label, mean_value)
#
#
#      #  #for stat, stat_spec in zip(stats, episode_stat_specs):
#      #  #  if stat_spec.name == 'summary_name' or len(stat) <= 0:
#      #  #    continue
#
#      #  #  for idx, summary_name in enumerate(unique_summary_names):
#      #  #    add_to_summary = unique_summary_name_idx == idx
#      #  #    stat_masked = tf.boolean_mask(stat, add_to_summary)
#      #  #    label = f'{summary_name.numpy().decode()}/mean_{stat_spec.name}'
#      #  #    if len(stat_masked) > 0:  # pylint: disable=g-explicit-length-test
#      #  #      log_mean_value(stat_masked, label=label)
#
#      logs = minimize(it)
#
#      #if (config.enable_learner_logging == 1 and
#      #    iterations % config.log_frequency == 0):
#      #  for per_replica_logs in logs:
#      #    assert len(log_keys) == len(per_replica_logs)
#      #    for key, value in zip(log_keys, per_replica_logs):
#      #      try:
#      #        values_to_log[key].append(value.numpy())
#      #      except AttributeError:
#      #        values_to_log[key].extend(
#      #            x.numpy()
#      #            for x in training_strategy.experimental_local_results(value))
#      #
#      #  log_future.result()  # Raise exception if any occurred in logging.
#      #  log_future = executor.submit(log, iterations)
#
#  #manager.save()

class Learner:
  def __init__(self, config , agent, create_optimizer, env_descriptor, pretraining=False):
    self.cfg = config
    self.agent = agent
    self.create_optimizer_fn = create_optimizer
    self.pretraining = pretraining
    self.env_descriptor = env_descriptor
    self.build()

  #def tf_function(func):
  #  def wrapper(self, func):
  #    return noop_decorator if self.cfg.lrn.debug else tf.function
  #  return wrapper
  def initialize_network(self):
    initial_agent_state = self.agent.initial_state(1)
    if self.cfg.lrn.debug:
      logging.info('initial state:\n{}'.format(initial_agent_state))

    agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)

    zero_observation = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), self.cfg.specs.observation)
    zero_action = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), self.cfg.specs.action)

    zero_initial_args = [self.encode(zero_observation)]
    zero_recurrent_args = [self.encode(initial_agent_state), self.encode(zero_action)]
     
    def create_variables(initial_args, recurrent_args, pretraining_args):
      self.agent.initial_inference(*map(self.decode, initial_args))
      self.agent.recurrent_inference(*map(self.decode, recurrent_args))
      if pretraining_args is not None:
        self.agent.pretraining_loss(*map(self.decode, pretraining_args))
    
    if self.pretraining:
      zero_pretraining = tf.nest.map_structure(
        lambda s: tf.zeros([1] + list(s.shape), s.dtype), self.cfg.specs.pretraining)
      zero_pretraining_args = [self.encode(zero_pretraining)]
      logging.info('zero pretraining args:\n{}'.format(zero_pretraining_args))
    else:
      zero_pretraining_args = None

    # This complicates BatchNormalization, can't use it.
    create_variables(zero_initial_args, zero_recurrent_args, zero_pretraining_args)

  def build(self):
    # Build stuff
    self. parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(self.env_descriptor.action_space)
    settings = utils.init_learner(self.cfg.lrn.num_training_tpus)
    strategy, inference_devices, training_strategy, self.encode, self.decode = settings

    # Build optimizer
    self.optimizer, self.learning_rate_fn = self.create_optimizer_fn(self.cfg.lrn.total_iterations)
    self.iterations = self.optimizer.iterations
    self.optimizer._create_hypers()
    self.optimizer._create_slots(
        self.agent.get_trainable_variables(pretraining=self.pretraining))
    # Build logs
    self.logdir = os.path.join(self.cfg.lrn.logdir, 'learner')
    self.saved_model_dir = os.path.join(self.logdir, 'saved_model')
    self.summary_writer = tf.summary.create_file_writer(
      self.logdir, flush_millis=self.cfg.lrn.flush_learner_log_every_n_s * 1000, max_queue=int(1E6))
    # Build dataset
    DL = DataLoader(self.cfg)
    self.dataset = DL.makeGenerator()
    self.iterator = iter(self.dataset)
    # Build checkpoint manager
    self.ckpt = tf.train.Checkpoint(agent=self.agent, optimizer=self.optimizer)
    self.manager = tf.train.CheckpointManager(self.ckpt, self.logdir, max_to_keep=10, keep_checkpoint_every_n_hours=6)

  @tf.function
  def compute_loss(self, *args, **kwargs):
    if self.pretraining:
      return compute_pretrain_loss(self.cfg.lrn, *args, **kwargs)
    else:
      return compute_loss(self.cfg.lrn, *args, **kwargs)

  @tf.function
  def minimize(self, data):
    args = tf.nest.pack_sequence_as(self.cfg.specs.weighted_replay_buffer, self.decode(data, data))
    with tf.GradientTape() as tape:
      loss, logs = self.compute_loss(self.parametric_action_distribution, self.agent, *args)
    grads = tape.gradient(loss, self.agent.get_trainable_variables(pretraining=self.pretraining))
    grads = grads
    if self.cfg.lrn.gradient_norm_clip > 0.:
      grads, _ = tf.clip_by_global_norm(grads, self.cfg.lrn.gradient_norm_clip)
    self.optimizer.apply_gradients(
        zip(grads, self.agent.get_trainable_variables(pretraining=self.pretraining)))
    return logs

  def tbd(self):
    logging.info('--------------------------')
    logging.info('TRAINABLE VARIABLES')
    for v in self.agent.get_trainable_variables(pretraining=self.pretraining):
      logging.info('{}: {} | {}'.format(v.name, v.shape, v.dtype))
    logging.info('--------------------------')

  def resumeFromCheckpoint(self):
    # Continuing a run from an intermediate checkpoint.  On this path, we do not
    # need to read `init_checkpoint`.
    if self.manager.latest_checkpoint:
      logging.info('Restoring checkpoint: %s', self.manager.latest_checkpoint)
      self.ckpt.restore(self.manager.latest_checkpoint).assert_consumed()
      self.last_ckpt_time = time.time()
      # Also properly reset iterations.
      self.iterations = self.optimizer.iterations
    else:
      self.last_ckpt_time = 0  # Force checkpointing of the initial model.
      # If there is a checkpoint from pre-training specified, load it now.
      # Note that we only need to do this if we are not already restoring a
      # checkpoint from the actual training.
      if self.cfg.lrn.init_checkpoint is not None:
        logging.info('Loading initial checkpoint from %s ...',self.cfg.lrn.init_checkpoint)
        # We don't want to restore the optimizer from pretraining
        ckpt_without_optimizer = tf.train.Checkpoint(agent=self.agent)
        # Loading checkpoints from independent pre-training might miss, for
        # example, optimizer weights (or have used different optimizers), and
        # might also not have fully instantiated all network parts (e.g. the
        # "core"-recurrence).
        # We still want to catch cases where nothing at all matches, but can not
        # do anything stricter here.
        ckpt_without_optimizer.restore(self.cfg.lrn.init_checkpoint).assert_nontrivial_match()
        logging.info('Finished loading the initial checkpoint.')

  def makeCheckpoint(self):
    # Save checkpoint.
    current_time = time.time()
    print(current_time - self.last_ckpt_time)
    print(self.cfg.lrn.save_checkpoint_secs)
    if current_time - self.last_ckpt_time >= self.cfg.lrn.save_checkpoint_secs:
      logging.info("Creating new checkpoint")
      self.manager.save()
      if self.cfg.lrn.export_agent:
        # We also export the agent as a SavedModel to be used for inference.
        network.export_agent_for_initial_inference(
            agent=self.agent,
            model_dir=os.path.join(self.saved_model_dir, 'initial_inference'))
        network.export_agent_for_recurrent_inference(
            agent=self.agent,
            model_dir=os.path.join(self.saved_model_dir, 'recurrent_inference'))
      self.last_ckpt_time = current_time

  def train(self):
    self.initialize_network()
    self.resumeFromCheckpoint()
    with self.summary_writer.as_default():
      while self.iterations < self.cfg.lrn.total_iterations:
        self.makeCheckpoint()
        data = next(self.iterator)
        logs = self.minimize(data)
        print(self.iterations)

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