# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""Actor command line flags."""

from absl import flags
from config_actor import ActorConfig

ACTOR_ID = flags.DEFINE_integer('actor_id', 0, 'Actor id.')
USE_SOFTMAX_FOR_TARGET = flags.DEFINE_integer(
    'use_softmax_for_target', 0,
    'If True (1), use a softmax for the child_visit count distribution that '
    'is used as a target for the policy.')
NUM_TEST_ACTORS = flags.DEFINE_integer(
    'num_test_actors', 2, 'Number of actors that are used for testing.')
NUM_ACTORS_WITH_SUMMARIES = flags.DEFINE_integer(
    'num_actors_with_summaries', 1,
    'Number of actors that will log debug/profiling TF '
    'summaries.')
ACTOR_LOG_FREQUENCY = flags.DEFINE_integer('actor_log_frequency', 10,
                                           'in number of training steps')
MCTS_VIS_FILE = flags.DEFINE_string(
    'mcts_vis_file', None, 'File in which to log the mcts visualizations.')
FLAG_FILE = flags.DEFINE_string('flag_file', None,
                                'File in which to log the parameters.')
ENABLE_ACTOR_LOGGING = flags.DEFINE_boolean('enable_actor_logging', True,
                                            'Verbose logging for the actor.')
MAX_NUM_ACTION_EXPANSION = flags.DEFINE_integer(
    'max_num_action_expansion', 0,
    'Maximum number of new nodes for a node expansion. 0 for no limit. '
    'This is important for the full vocabulary.')
ACTOR_ENQUEUE_EVERY = flags.DEFINE_integer(
    'actor_enqueue_every', 0,
    'After how many steps the actor enqueues samples. 0 for at episode end.')
ACTOR_SKIP = flags.DEFINE_integer('actor_skip', 0,
                                  'How many target samples the actor skips.')

FLAGS = flags.FLAGS

def actor_config_from_flags():
    """Returns the actor's config based on command line flags."""
    return ActorConfig(
        actor_id = FLAGS.actor_id,
        use_softmax_for_target=FLAGS.use_softmax_for_target,
        num_test_actors=FLAGS.num_test_actors,
        num_actors_with_summaries=FLAGS.num_actors_with_summaries,
        actor_log_frequency=FLAGS.actor_log_frequency,
        mcts_vis_file=FLAGS.mcts_vis_file,
        flag_file=FLAGS.flag_file,
        enable_actor_logging=FLAGS.enable_actor_logging,
        max_num_action_expansion=FLAGS.max_num_action_expansion,
        actor_enqueue_every=FLAGS.actor_enqueue_every,
        actor_skip=FLAGS.actor_skip)

def actor_config_from_file(flag_dict):
  return ActorConfig(**flag_dict)