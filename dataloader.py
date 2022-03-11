import BufferUtils as utils
import tensorflow as tf
import numpy as np
import pathlib
import pickle
from absl import logging

class DataLoader:
  def __init__(self, cfg):
    # Load
    self.cfg = cfg
    # Create replay buffer
    self.replay_buffer = utils.PrioritizedReplay(
        self.cfg.lrn.replay_buffer_size,
        self.cfg.specs.replay_buffer,
        self.cfg.lrn.importance_sampling_exponent,
    )
    self.cache = []

  def sample(self):
    indices, weights, replays = self.replay_buffer.sample(self.cfg.lrn.batch_size, self.cfg.lrn.priority_sampling_exponent)
    if self.cfg.lrn.replay_buffer_update_priority_after_sampling_value >= 0.:
      self.replay_buffer.update_priorities(
          indices,
          tf.convert_to_tensor(
              np.ones(indices.shape) *
              self.cfg.lrn.replay_buffer_update_priority_after_sampling_value,
              dtype=tf.float32))
    data = (weights, *replays)
    return tf.nest.flatten(data)
  
  def loadBatchesGen(self):
    directory = pathlib.Path(self.cfg.mz.episode_dir).expanduser()
    logging.info("Loading dataset from folder: %s",directory)
    cache = []
    while True:
      for filename in directory.glob('*.pkl'):
        if filename not in cache:
          try:
            with filename.open('rb') as f:
              batch = pickle.load(f)
          except Exception as e:
            logging.warn(f'Could not load episode: {e}')
            continue
          cache.append(filename)
          priorities, *samples = batch
          self.replay_buffer.insert(tuple(samples), priorities)
          if self.replay_buffer.num_inserted <= self.cfg.lrn.replay_buffer_size:
            logging.info('waiting for replay buffer to fill. Status:%d / %d', self.replay_buffer.num_inserted, self.cfg.lrn.replay_buffer_size)
          elif self.cfg.lrn.debug:
            logging.info('Replay buffer filled with %d samples', self.replay_buffer.num_inserted)
      yield tuple(self.sample())
  
  def makeGenerator(self):
      return tf.data.Dataset.from_generator(self.loadBatchesGen, output_signature=self.cfg.specs.batch)