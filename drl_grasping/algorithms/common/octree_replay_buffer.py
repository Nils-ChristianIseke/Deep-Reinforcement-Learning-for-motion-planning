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

# This module adds a monkey patch to ReplayBuffer, such that octrees are
# directly supported and there is no extra RAM -> VRAM -> RAM overhead

# Note: needs to be included before `from stable_baselines3.common.buffers import ReplayBuffer` in the module that uses this

from stable_baselines3.common.buffers import ReplayBuffer

import numpy as np
import torch as th
from typing import Optional, Union, Dict
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gym import spaces

import ocnn


def preprocess_stacked_octree_batch(observation: th.Tensor, device, separate_batches: bool = True) -> Dict[str, th.Tensor]:
    # Note: Primordial magic is happening here,
    #       but there's no reason to tremble in fear.
    #       For your own good don't question it too much,
    #       it's just an optimised stacked octree batch...

    if not separate_batches:
        octrees = []
        for octree in observation.reshape(-1, observation.shape[-1]):
            # Get original octree size
            octree_size = np.frombuffer(buffer=octree[-4:],
                                        dtype='uint32',
                                        count=1)
            # Convert to tensor and append to list
            octrees.append(th.from_numpy(octree[:octree_size[0]]))
        # Make batch out of tensor (consisting of n-stacked frames)
        octree_batch = ocnn.octree_batch(octrees)

        # Get number of auxiliary observations encoded as float32 and parse them
        n_aux_obs_f32 = int(np.frombuffer(buffer=observation[0, 0, -8:-4],
                                          dtype='uint32',
                                          count=1))
        aux_obs = th.from_numpy(
            np.frombuffer(buffer=observation[:, :, -(4*n_aux_obs_f32+8):-8].reshape(-1),
                          dtype='float32',
                          count=n_aux_obs_f32*observation.shape[0]*observation.shape[1]).reshape(observation.shape[:2] + (n_aux_obs_f32,)))

        return {'octree': octree_batch.to(device),
                'aux_obs': aux_obs.to(device)}

    else:
        octree_batches = []

        for octree_batch in np.split(observation, observation.shape[1], axis=1):
            octrees = []
            for octree in octree_batch:
                # Get original octree size
                octree_size = np.frombuffer(buffer=octree[-4:],
                                            dtype='uint32',
                                            count=1)
                # Convert to tensor and append to list
                octrees.append(th.from_numpy(octree[:octree_size[0]]))
            # Make batch out of tensor (consisting of one stack)
            octree_batches.append(ocnn.octree_batch(octrees).to(device))

        # Get number of auxiliary observations encoded as float32 and parse them
        n_aux_obs_f32 = int(np.frombuffer(buffer=observation[0, 0, -8:-4],
                                          dtype='uint32',
                                          count=1))
        aux_obs = th.from_numpy(
            np.frombuffer(buffer=observation[:, :, -(4*n_aux_obs_f32+8):-8].reshape(-1),
                          dtype='float32',
                          count=n_aux_obs_f32*observation.shape[0]*observation.shape[1]).reshape(observation.shape[:2] + (n_aux_obs_f32,)))

        return {'octree': octree_batches,
                'aux_obs': aux_obs.to(device)}


__old__init__ = ReplayBuffer.__init__
__old_get_samples__ = ReplayBuffer._get_samples


def __init___with_checking_for_octree(self,
                                      buffer_size: int,
                                      observation_space: spaces.Space,
                                      action_space: spaces.Space,
                                      device: Union[th.device, str] = "cpu",
                                      n_envs: int = 1,
                                      optimize_memory_usage: bool = False,
                                      separate_networks_for_stacks: bool = True):
    __old__init__(self,
                  buffer_size=buffer_size,
                  observation_space=observation_space,
                  action_space=action_space,
                  device=device,
                  n_envs=n_envs,
                  optimize_memory_usage=optimize_memory_usage)

    # Determine if octrees are used
    # Note: This is not 100% reliable as there could be other observations that do the same (outside of this repo)
    self.contains_octree_obs = False
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 2:
        if np.uint8 == observation_space.dtype and \
            np.all(0 == observation_space.low) and \
                np.all(255 == observation_space.high):
            self.contains_octree_obs = True
            self._separate_networks_for_stacks = separate_networks_for_stacks


def _get_samples_with_support_for_octree(self,
                                         batch_inds: np.ndarray,
                                         env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

    if not self.contains_octree_obs:
        return __old_get_samples__(self, batch_inds=batch_inds, env=env)

    # Current observations
    obs = self.observations[batch_inds, 0, :]
    obs = preprocess_stacked_octree_batch(obs, self.device, separate_batches=self._separate_networks_for_stacks)

    # Next observations
    if self.optimize_memory_usage:
        next_obs = self.observations[(
            batch_inds + 1) % self.buffer_size, 0, :]
    else:
        next_obs = self.next_observations[batch_inds, 0, :]
    next_obs = preprocess_stacked_octree_batch(next_obs, self.device, separate_batches=self._separate_networks_for_stacks)

    return ReplayBufferSamples(
        observations=obs,
        actions=self.to_torch(self.actions[batch_inds, 0, :]),
        next_observations=next_obs,
        dones=self.to_torch(self.dones[batch_inds]),
        rewards=self.to_torch(self._normalize_reward(
            self.rewards[batch_inds], env)),
    )


ReplayBuffer.__init__ = __init___with_checking_for_octree
ReplayBuffer._get_samples = _get_samples_with_support_for_octree
