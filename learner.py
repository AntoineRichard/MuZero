from absl import logging
import tensorflow as tf
import collections
import copy
import time
import os
import distributions
from losses import compute_loss, compute_pretrain_loss
import BufferUtils as utils
from dataloader import DataLoader
import core_network as network

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

    create_variables(zero_initial_args, zero_recurrent_args, zero_pretraining_args)
    logging.info('--------------------------')
    logging.info('TRAINABLE VARIABLES')
    for v in self.agent.get_trainable_variables(pretraining=self.pretraining):
      logging.info('{}: {} | {}'.format(v.name, v.shape, v.dtype))
    logging.info('--------------------------')


  def build(self):
    # Build stuff
    self.parametric_action_distribution = distributions.get_parametric_distribution_for_action_space(self.env_descriptor.action_space)
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
    self.values_to_log = collections.defaultdict(lambda: [])
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
    if current_time - self.last_ckpt_time >= self.cfg.lrn.save_checkpoint_secs:
      logging.info("Creating new checkpoint.")
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
        self.writeToSummary(logs)

  @staticmethod
  def log_mean_value(values, label):
    mean_value = tf.reduce_mean(tf.cast(values, tf.float32))
    tf.summary.scalar(label, mean_value)
      
  def writeToSummary(self, log):
    if (self.cfg.lrn.enable_learner_logging == 1 and
        self.iterations % self.cfg.lrn.log_frequency == 0):
      for key in log.keys():
        try:
          self.values_to_log[key].append(log[key].numpy())
        except AttributeError:
          self.values_to_log[key].extend(x.numpy() for x in log[key])

      self.summary_writer.set_as_default()
      tf.summary.experimental.set_step(self.iterations)
      # log data from the current minibatch
      for key, values in copy.deepcopy(self.values_to_log).items():
        if values:
          tf.summary.scalar(key, values[-1])  # could also take mean
      self.values_to_log.clear()
      logging.info("Step %d: writting to summary.", self.iterations)
      tf.summary.scalar('learning_rate', self.learning_rate_fn(self.iterations))