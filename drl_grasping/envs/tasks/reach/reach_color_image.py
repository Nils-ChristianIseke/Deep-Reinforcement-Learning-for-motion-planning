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


from drl_grasping.envs.tasks.reach import Reach
from drl_grasping.perception import CameraSubscriber
from gym_ignition.utils.typing import Observation
from gym_ignition.utils.typing import ObservationSpace
from typing import Tuple
import abc
import gym
import numpy as np


class ReachColorImage(Reach, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _camera_enable: bool = True
    _camera_type: str = 'camera'
    _camera_width: int = 128
    _camera_height: int = 128
    _camera_update_rate: int = 10
    _camera_horizontal_fov: float = 1.0
    _camera_vertical_fov: float = 1.0
    _camera_position: Tuple[float, float, float] = (1.1, -0.75, 0.45)
    _camera_quat_xyzw: Tuple[float, float,
                             float, float] = (-0.0402991, -0.0166924, 0.9230002, 0.3823192)
    _camera_ros2_bridge_color: bool = True

    def __init__(self,
                 agent_rate: float,
                 robot_model: str,
                 restrict_position_goal_to_workspace: bool,
                 sparse_reward: bool,
                 act_quick_reward: float,
                 required_accuracy: float,
                 verbose: bool,
                 **kwargs):

        # Initialize the Task base class
        Reach.__init__(self,
                       agent_rate=agent_rate,
                       robot_model=robot_model,
                       restrict_position_goal_to_workspace=restrict_position_goal_to_workspace,
                       sparse_reward=sparse_reward,
                       act_quick_reward=act_quick_reward,
                       required_accuracy=required_accuracy,
                       verbose=verbose,
                       **kwargs)

        # Perception (RGB camera)
        self.camera_sub = CameraSubscriber(topic=f'/{self._camera_type}',
                                           is_point_cloud=False,
                                           node_name=f'drl_grasping_rgb_camera_sub_{self.id}')

    def create_observation_space(self) -> ObservationSpace:

        # 0:3*height*width - rgb image
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._camera_height,
                                     self._camera_width, 3),
                              dtype=np.uint8)

    def get_observation(self) -> Observation:

        # Get the latest image
        image = self.camera_sub.get_observation()

        # Reshape and create the observation
        color_image = np.array(image.data, dtype=np.uint8).reshape(self._camera_height,
                                                                   self._camera_width, 3)

        observation = Observation(color_image)

        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation
