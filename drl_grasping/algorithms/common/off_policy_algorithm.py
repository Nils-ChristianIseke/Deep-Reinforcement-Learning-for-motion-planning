## BSD 3-Clause License
##
## Copyright (c) 2021, Andrej Orsula
## All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:

## 1. Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
##
## 3. Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This module adds a monkey patch to OffPolicyAlgorithm `collect_rollouts` function
# Path that sets done to False if the episode was truncated

# Note: needs to be included before `from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm` in the module that uses this

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from typing import Any, Dict, List

import numpy as np


# Note: Import monkey patch of ReplayBuffer before stable_baselines3 ReplayBuffer
from drl_grasping.algorithms.common import octree_replay_buffer

from stable_baselines3.common.buffers import ReplayBuffer


def _store_transition_not_done_if_truncated(
    self,
    replay_buffer: ReplayBuffer,
    buffer_action: np.ndarray,
    new_obs: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    infos: List[Dict[str, Any]],
) -> None:
    """
    Store transition in the replay buffer.
    We store the normalized action and the unnormalized observation.
    It also handles terminal observations (because VecEnv resets automatically).

    :param replay_buffer: Replay buffer object where to store the transition.
    :param buffer_action: normalized action
    :param new_obs: next observation in the current episode
        or first observation of the episode (when done is True)
    :param reward: reward for the current transition
    :param done: Termination signal
    :param infos: List of additional information about the transition.
        It contains the terminal observations.
    """
    # Store only the unnormalized version
    if self._vec_normalize_env is not None:
        new_obs_ = self._vec_normalize_env.get_original_obs()
        reward_ = self._vec_normalize_env.get_original_reward()
    else:
        # Avoid changing the original ones
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

    # As the VecEnv resets automatically, new_obs is already the
    # first observation of the next episode
    if done and infos[0].get("terminal_observation") is not None:
        next_obs = infos[0]["terminal_observation"]
        # VecNormalize normalizes the terminal observation
        if self._vec_normalize_env is not None:
            next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
    else:
        next_obs = new_obs_

    # NOTE: The monkey patch is inside the following block of code
    done_ = np.array([False]) if infos[0].get(
        "TimeLimit.truncated", False) else done
    replay_buffer.add(self._last_original_obs, next_obs,
                      buffer_action, reward_, done_)
    # replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done)

    self._last_obs = new_obs
    # Save the unnormalized observation
    if self._vec_normalize_env is not None:
        self._last_original_obs = new_obs_


# So ugly, lul
def _setup_model_with_separate_octree_batches_for_stacks(self) -> None:
    self._setup_lr_schedule()
    self.set_random_seed(self.seed)
    if 'separate_networks_for_stacks' in self.policy_kwargs:
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            separate_networks_for_stacks=self.policy_kwargs['separate_networks_for_stacks'],
        )
    else:
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
    self.policy = self.policy_class(  # pytype:disable=not-instantiable
        self.observation_space,
        self.action_space,
        self.lr_schedule,
        **self.policy_kwargs,  # pytype:disable=not-instantiable
    )
    self.policy = self.policy.to(self.device)

    # Convert train freq parameter to TrainFreq object
    self._convert_train_freq()


# OffPolicyAlgorithm._store_transition = _store_transition_not_done_if_truncated
OffPolicyAlgorithm._setup_model = _setup_model_with_separate_octree_batches_for_stacks
