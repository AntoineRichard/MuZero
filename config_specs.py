import dataclasses
import tensorflow as tf

@dataclasses.dataclass
class SpecsConfig:
  """Config for the tensor specs."""
  batch: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  action: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  target: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  agent_state: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  observation: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  replay_buffer: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  initial_inference: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  recurrent_inference: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")
  weighted_replay_buffer: tf.TensorSpec = tf.TensorSpec([],tf.float32, "none")