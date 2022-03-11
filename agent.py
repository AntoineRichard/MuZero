from absl import logging
import tensorflow as tf
import numpy as np
import random
import collections
import pickle
import os
import datetime

import core as mzcore
import string

class Agent:
    def __init__(self, createEnvFn, cfg, is_training=True):
        self.createEnvironment = createEnvFn
        self.cfg = cfg
        self.is_training_actor = is_training
        self.init_queues()

    @staticmethod
    def generateRandomString():
        return "".join([random.choice(list(string.ascii_letters)) for i in range(6)])

    def init_queues(self):
        self.save_dir = os.path.join(self.cfg.lrn.logdir,'episodes','actor_'+str(self.cfg.act.actor_id))
        os.makedirs(self.save_dir,exist_ok=True)
        self.actor_step = tf.Variable(0, dtype=tf.int64)
        self.batch_queue = collections.deque()

    def create_training_samples(self, episode, start_idx=0):
        start_idx += random.choice(range(self.cfg.act.actor_skip + 1))
        for i in range(start_idx, len(episode.history), self.cfg.act.actor_skip + 1):
            target = episode.make_target(
                state_index=i,
                num_unroll_steps=self.cfg.mz.num_unroll_steps,
                td_steps=self.cfg.mz.td_steps,
                rewards=episode.rewards,
                policy_distributions=episode.child_visits,
                discount=episode.discount,
                value_approximations=episode.root_values)
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
                    episode.history_range(i, i + self.cfg.mz.num_unroll_steps)),
            ) + tuple(map(lambda x: tf.cast(tf.stack(x), tf.float32), target))
            self.batch_queue.append(sample)
        if self.cfg.act.enable_actor_logging:
            logging.info(
                'Added %d samples to the batch_queue. Size: %d of needed %d',
                len(episode.history) - start_idx, len(self.batch_queue),
                self.cfg.lrn.batch_size)

    def save_queue_to_file(self):
        while len(self.batch_queue) >= self.cfg.lrn.batch_size:
            date = datetime.datetime.now().strftime('%d-%m-%Y_%HH%MM%SS')
            filename = os.path.join(self.save_dir, date +'_'+self.generateRandomString()+'_episode.pkl')
            batch = [
                self.batch_queue.popleft()
                for _ in range(self.cfg.lrn.batch_size)
            ]
            flat_batch = [tf.nest.flatten(b) for b in batch]
            stacked_batch = list(map(tf.stack, zip(*flat_batch)))
            structured_batch = tf.nest.pack_sequence_as(
                batch[0], stacked_batch)
            with open(filename,'wb') as f:
                pickle.dump(structured_batch, f)
            if self.cfg.act.enable_actor_logging:
                logging.info('saved batch of size %d to file %s.',
                           len(batch), filename)

    def playEpisode(self, initialInference, recurrentInference, trn_step):
        episode = self.cfg.mz.new_episode(self.createEnvironment(0,self.is_training_actor))
        legal_actions_fn = episode.legal_actions
        last_enqueued_idx = 0
        while (not episode.terminal() and len(episode.history) < self.cfg.mz.max_moves):
            current_observation = episode.make_image(-1)
            current_observation = np.expand_dims(current_observation,0)
            initial_inference_output = initialInference(current_observation)
            initial_inference_output = tf.nest.map_structure(lambda t: t.numpy(), initial_inference_output)
            root = mzcore.prepare_root_node(self.cfg.mz, legal_actions_fn(), initial_inference_output)
            mzcore.run_mcts(self.cfg.mz, root, episode.action_history(), legal_actions_fn,
              recurrentInference, episode.visualize_mcts)
            action = mzcore.select_action(self.cfg.mz, len(episode.history), root, train_step=trn_step,
              use_softmax=self.cfg.mz.use_softmax_for_action_selection, is_training=False)
            try:
                episode.apply(action=action, training_steps=trn_step)
            except mzcore.RLEnvironmentError as env_error:
                logging.warning('Environment failed: %s', str(env_error))
                episode.failed = True
                break
            episode.store_search_statistics(root, use_softmax=(self.cfg.act.use_softmax_for_target == 1))
            self.actor_step.assign_add(delta=1)
            if self.is_training_actor and self.cfg.act.actor_enqueue_every > 0 and (len(episode.history) - last_enqueued_idx) >= self.cfg.act.actor_enqueue_every:
                self.create_training_samples(episode, start_idx=last_enqueued_idx)
                last_enqueued_idx = len(episode.history)
        self.create_training_samples(episode, start_idx=last_enqueued_idx)
        self.save_queue_to_file()