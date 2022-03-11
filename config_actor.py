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

"""Learner config."""

import dataclasses

@dataclasses.dataclass
class ActorConfig:
  """Config for the actor."""
  # Actor id.
  actor_id: int = 0
  # If True (1), use a softmax for the child_visit count distribution that 
  # is used as a target for the policy.
  use_softmax_for_target: int = 0
  # Number of actors that are used for testing.
  num_test_actors: int = 2
  # Number of actors that will log debug/profiling TF summaries.
  num_actors_with_summaries: int = 1
  # in number of training steps.
  actor_log_frequency: int = 10
  # File in which to log the mcts visualizations.
  mcts_vis_file: str = None
  # File in which to log the parameters.
  flag_file: str = None
  # Verbose logging for the actor.
  enable_actor_logging: bool = True
  # Maximum number of new nodes for a node expansion. 0 for no limit.
  max_num_action_expansion: int = 0
  # After how many steps the actor enqueues samples. 0 for at episode end.
  actor_enqueue_every: int = 0
  # How many target samples the actor skips.
  actor_skip: int = 0
