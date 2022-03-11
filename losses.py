from absl import logging
from config_learner import LearnerConfig
import numpy as np
import tensorflow as tf
import core
import collections

def scale_gradient(t, scale):
  return scale * t + (1 - scale) * tf.stop_gradient(t)

def compute_pretrain_loss(config: LearnerConfig,
                          parametric_action_distribution, agent,
                          importance_weights, *sample):

  if config.debug and np.random.rand() < 1 / 50:
    logging.info('-------------------')
    logging.info('pretrain sample:')
    logging.info(sample)
    logging.info('-------------------')

  loss, pretrain_logs = agent.pretraining_loss(sample)
  loss = loss * importance_weights  # importance sampling correction
  mean_loss = tf.math.divide_no_nan(
      tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

  if config.weight_decay > 0.:
    l2_loss = config.weight_decay * sum(
        tf.nn.l2_loss(v)
        for v in agent.get_trainable_variables(pretraining=True))
  else:
    l2_loss = mean_loss * 0.

  total_loss = mean_loss + l2_loss

  #del log_keys[:]
  #log_values = []

  # logging

  #def log(key, value):
  #  # this is a python op so it happens only when this tf.function is compiled
  #  log_keys.append(key)
  #  # this is a TF op
  #  log_values.append(value)

  #log('pretrain/losses/sample', mean_loss)
  #log('pretrain/losses/weight_decay', l2_loss)
  #log('pretrain/losses/total', total_loss)
  #if pretrain_logs is not None:
  #  for ptk, ptv in pretrain_logs.items():
  #    log('pretrain/{}'.format(ptk), tf.reduce_mean(ptv))

  return total_loss#, log_values


def compute_loss(config: LearnerConfig,
                 parametric_action_distribution, agent, importance_weights,
                 observation, history, target_value_mask, target_reward_mask,
                 target_policy_mask, target_value, target_reward,
                 target_policy):

  # initial step
  output = agent.initial_inference(observation)
  predictions = [
      core.Prediction(
          gradient_scale=1.0,
          value=output.value,
          value_logits=output.value_logits,
          reward=output.reward,
          reward_logits=output.reward_logits,
          policy_logits=output.policy_logits,
      )
  ]
  # recurrent steps
  num_recurrent_steps = history.shape[-1]
  for rstep in range(num_recurrent_steps):
    hidden_state_gradient_scale = 1.0 if rstep == 0 else 0.5
    output = agent.recurrent_inference(
        scale_gradient(output.hidden_state, hidden_state_gradient_scale),
        history[:, rstep],
    )
    predictions.append(
        core.Prediction(
            gradient_scale=1.0 / num_recurrent_steps,
            value=output.value,
            value_logits=output.value_logits,
            reward=output.reward,
            reward_logits=output.reward_logits,
            policy_logits=output.policy_logits,
        ))

  num_target_steps = target_value.shape[-1]
  assert len(predictions) == num_target_steps, (
      'There should be as many predictions ({}) as targets ({})'.format(
          len(predictions), num_target_steps))
  masks = {
      'value': target_value_mask,
      'reward': target_reward_mask,
      'policy': target_policy_mask,
      'action': target_policy_mask,
  }

  def name_to_mask(name):
    return next(k for k in masks if k in name)

  # This is more rigorous than the MuZero paper.
  gradient_scales = {
      k: tf.math.divide(1.0, tf.maximum(tf.reduce_sum(m[:, 1:], -1), 1.0))
      for k, m in masks.items()
  }
  gradient_scales = {
      k: [tf.ones_like(s)] + [s] * (num_target_steps - 1)
      for k, s in gradient_scales.items()
  }

  target_reward_encoded, target_value_encoded = (tf.reshape(
      enc.encode(tf.reshape(v, (-1,))),
      (-1, num_target_steps,
       enc.num_steps)) for enc, v in ((agent.reward_encoder, target_reward),
                                      (agent.value_encoder, target_value)))

  # Accumulators over time steps.
  accs = collections.defaultdict(list)
  for tstep, prediction in enumerate(predictions):
    accs['value_loss'].append(
        scale_gradient(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.value_logits,
                labels=target_value_encoded[:, tstep]),
            gradient_scales['value'][tstep]))
    accs['reward_loss'].append(
        scale_gradient(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction.reward_logits,
                labels=target_reward_encoded[:, tstep]),
            gradient_scales['reward'][tstep]))
    policy_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction.policy_logits, labels=target_policy[:, tstep])
    entropy_loss = -parametric_action_distribution.entropy(
        prediction.policy_logits) * config.policy_loss_entropy_regularizer
    accs['policy_loss'].append(
        scale_gradient(policy_loss + entropy_loss,
                       gradient_scales['policy'][tstep]))

    accs['value_diff'].append(
        tf.abs(tf.squeeze(prediction.value) - target_value[:, tstep]))
    accs['reward_diff'].append(
        tf.abs(tf.squeeze(prediction.reward) - target_reward[:, tstep]))
    accs['policy_acc'].append(
        tf.keras.metrics.categorical_accuracy(
            target_policy[:, tstep],
            tf.nn.softmax(prediction.policy_logits, axis=-1)))

    accs['value'].append(tf.squeeze(prediction.value))
    accs['reward'].append(tf.squeeze(prediction.reward))
    accs['action'].append(
        tf.cast(tf.argmax(prediction.policy_logits, -1), tf.float32))

    accs['target_value'].append(target_value[:, tstep])
    accs['target_reward'].append(target_reward[:, tstep])
    accs['target_action'].append(
        tf.cast(tf.argmax(target_policy[:, tstep], -1), tf.float32))

  accs = {k: tf.stack(v, -1) * masks[name_to_mask(k)] for k, v in accs.items()}

  if config.debug and np.random.rand() < 1 / 50:
    logging.info('-------------------')
    logging.info(observation)
    for k, v in accs.items():
      logging.info('{}:\n{}'.format(k, v))
    for k, v in masks.items():
      logging.info('mask {}:\n{}'.format(k, v))
    logging.info('history:\n{}'.format(history))
    logging.info('target_policy:\n{}'.format(target_policy))
    logging.info('importance_weights:\n{}'.format(importance_weights))
    logging.info('-------------------')

  loss = accs['value_loss'] + config.reward_loss_scaling * accs[
      'reward_loss'] + config.policy_loss_scaling * accs['policy_loss']
  loss = tf.reduce_sum(loss, -1)  # aggregating over time
  loss = loss * importance_weights  # importance sampling correction
  mean_loss = tf.math.divide_no_nan(
      tf.reduce_sum(loss), tf.reduce_sum(importance_weights))

  if config.weight_decay > 0.:
    l2_loss = config.weight_decay * sum(
        tf.nn.l2_loss(v)
        for v in agent.get_trainable_variables(pretraining=False))
  else:
    l2_loss = mean_loss * 0.

  mean_loss += l2_loss

  #print(num_target_steps)
  #tf.print(num_target_steps)
  #del log_keys[:]
  log_values = {}

  #logging

  def log(key, value):
    # this is a python op so it happens only when this tf.function is compiled
    #log_keys.append(key)
    # this is a TF op
    log_values[key] = value

  log('losses/total', mean_loss)
  log('losses/weight_decay', l2_loss)

  sum_accs = {k: tf.reduce_sum(a, -1) for k, a in accs.items()}
  sum_masks = {
      k: tf.maximum(tf.reduce_sum(m, -1), 1.) for k, m in masks.items()
  }

  def get_mean(k):
    return tf.reduce_mean(sum_accs[k] / sum_masks[name_to_mask(k)])

  log('prediction/value', get_mean('value'))
  log('prediction/reward', get_mean('reward'))
  log('prediction/policy', get_mean('action'))

  log('target/value', get_mean('target_value'))
  log('target/reward', get_mean('target_reward'))
  log('target/policy', get_mean('target_action'))

  log('losses/value', tf.reduce_mean(sum_accs['value_loss']))
  log('losses/reward', tf.reduce_mean(sum_accs['reward_loss']))
  log('losses/policy', tf.reduce_mean(sum_accs['policy_loss']))

  log('accuracy/value', -get_mean('value_diff'))
  log('accuracy/reward', -get_mean('reward_diff'))
  log('accuracy/policy', get_mean('policy_acc'))

  return mean_loss, log_values