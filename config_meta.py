import tensorflow as tf

import settings_actor
import settings_learner
import settings_muzero
from config_specs import SpecsConfig

def make_spec_from_gym_space(space, name):
    if space.dtype is not None:
        specs = tf.TensorSpec(space.shape, space.dtype, name)
    else:
        # This is a tuple space
        specs = tuple(
            make_spec_from_gym_space(s, '{}_{}'.format(name, idx))
            for idx, s in enumerate(space.spaces))
    return specs

class MetaConfig:
    def __init__(self, learner_from_flag=False, learner_from_file=None, actor_from_flag=False, actor_from_file=None, muzero_from_flag=False, muzero_from_file=None):
        # Args
        self.learner_from_flag = learner_from_flag
        self.learner_from_file = learner_from_file
        self.muzero_from_flag = muzero_from_flag
        self.muzero_from_file = muzero_from_file
        self.actor_from_flag = actor_from_flag
        self.actor_from_file = actor_from_file
        # Configs
        self.lrn = None
        self.act = None
        self.mz = None
        self.specs = SpecsConfig
        self.sanity_check() 
    
    def sanity_check(self):
        if self.learner_from_flag and (self.learner_from_file is not None):
            raise Exception("cannot use both flags and file.")
        if (not self.learner_from_flag) and (self.learner_from_file is None):
            raise Warning("No file or flags passed, using default parameters.")
        if self.muzero_from_flag and (self.muzero_from_file is not None):
            raise Exception("cannot use both flags and file.")
        if (not self.muzero_from_flag) and (self.muzero_from_file is None):
            raise Warning("No file or flags passed, using default parameters.")
        if self.actor_from_flag and (self.actor_from_file is not None):
            raise Exception("cannot use both flags and file.")
        if (not self.actor_from_flag) and (self.actor_from_file is None):
            raise Warning("No file or flags passed, using default parameters.")

    def build_meta_config(self, action_space_size, known_bounds=None):
        if self.learner_from_flag:
            self.lrn = settings_learner.learner_config_from_flags()
        elif self.learner_from_file:
            self.lrn = settings_learner.learner_config_from_file(self.learner_from_file)
        else:
            self.lrn = settings_learner.LearnerConfig()
        if self.actor_from_flag:
            self.act = settings_actor.actor_config_from_flags()
        elif self.actor_from_file:
            self.act = settings_actor.actor_config_from_file(self.actor_from_file)
        else:
            self.act = settings_actor.ActorConfig()
        if self.muzero_from_flag:
            self.mz = settings_muzero.muzero_config_from_flags(action_space_size,
                                                            known_bounds=known_bounds)
        elif self.muzero_from_file:
            self.mz = settings_muzero.muzero_config_from_flags(action_space_size,
                                                            self.muzero_from_file,
                                                            known_bounds=known_bounds)

    def make_specs(self, env_descriptor, agent):        
        num_target_steps = self.mz.num_unroll_steps + 1
        initial_agent_state = agent.initial_state(1)
        self.specs.observation = make_spec_from_gym_space(env_descriptor.observation_space, 'observation')
        self.specs.action = make_spec_from_gym_space(env_descriptor.action_space, 'action')
        self.specs.initial_inference = (self.specs.observation,)
        self.specs.target = (
            tf.TensorSpec([num_target_steps], tf.float32, 'value_mask'),
            tf.TensorSpec([num_target_steps], tf.float32, 'reward_mask'),
            tf.TensorSpec([num_target_steps], tf.float32, 'policy_mask'),
            tf.TensorSpec([num_target_steps], tf.float32, 'value'),
            tf.TensorSpec([num_target_steps], tf.float32, 'reward'),
            tf.TensorSpec([num_target_steps, env_descriptor.action_space.n],
                    tf.float32, 'policy'),
        )
        self.specs.replay_buffer = (self.specs.observation,
                tf.TensorSpec(env_descriptor.action_space.shape + (self.mz.num_unroll_steps,),
                env_descriptor.action_space.dtype, 'history'), *self.specs.target)
        self.specs.batch = (
            tf.TensorSpec([self.lrn.batch_size], tf.float32, 'priorities'),
            tf.TensorSpec([self.lrn.batch_size] + list(env_descriptor.observation_space.shape), tf.float32, 'observation'),
            tf.TensorSpec([self.lrn.batch_size] + list(env_descriptor.action_space.shape + (self.mz.num_unroll_steps,)), tf.int64, 'history'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps], tf.float32, 'value_mask'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps], tf.float32, 'reward_mask'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps], tf.float32, 'policy_mask'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps], tf.float32, 'value'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps], tf.float32, 'reward'),
            tf.TensorSpec([self.lrn.batch_size, num_target_steps, env_descriptor.action_space.n], tf.float32, 'policy'))
        self.specs.agent_state = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
        self.specs.recurrent_inference = (self.specs.agent_state, self.specs.action)
        self.specs.weighted_replay_buffer = (tf.TensorSpec([], tf.float32, 'importance_weights'), *self.specs.replay_buffer)
        

