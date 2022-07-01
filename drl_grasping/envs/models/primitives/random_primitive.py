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


from drl_grasping.envs.models.primitives import Box, Cylinder, Sphere
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils import misc
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List, Union
import numpy as np


class RandomPrimitive(model_wrapper.ModelWrapper):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'primitive',
                 use_specific_primitive: Union[str, None] = None,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 static: bool = False,
                 collision: bool = True,
                 visual: bool = True,
                 gui_only: bool = False,
                 np_random=None):

        if np_random is None:
            np_random = np.random.default_rng()

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Create SDF string for the model
        sdf = self.get_sdf(model_name=model_name,
                           use_specific_primitive=use_specific_primitive,
                           static=static,
                           collision=collision,
                           visual=visual,
                           gui_only=gui_only,
                           np_random=np_random)

        # Convert it into a file
        sdf_file = misc.string_to_file(sdf)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(sdf_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError('Failed to insert ' + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)

    @classmethod
    def get_sdf(self,
                model_name: str,
                use_specific_primitive: Union[str, None],
                static: bool,
                collision: bool,
                visual: bool,
                gui_only: bool,
                np_random) -> str:

        if use_specific_primitive is not None:
            primitive = use_specific_primitive
        else:
            primitive = np_random.choice(['box', 'cylinder', 'sphere'])

        mass = np_random.uniform(0.05, 0.25)
        friction = np_random.uniform(0.75, 1.5)
        color = list(np_random.uniform(0.0, 1.0, (3,)))
        color.append(1.0)

        if 'box' == primitive:
            return Box.get_sdf(model_name=model_name,
                               size=list(np_random.uniform(0.04, 0.06, (3,))),
                               mass=mass,
                               static=static,
                               collision=collision,
                               friction=friction,
                               visual=visual,
                               gui_only=gui_only,
                               color=color)
        elif 'cylinder' == primitive:
            return Cylinder.get_sdf(model_name=model_name,
                                    radius=np_random.uniform(0.01, 0.0375),
                                    length=np_random.uniform(0.025, 0.05),
                                    mass=mass,
                                    static=static,
                                    collision=collision,
                                    friction=friction,
                                    visual=visual,
                                    gui_only=gui_only,
                                    color=color)
        elif 'sphere' == primitive:
            return Sphere.get_sdf(model_name=model_name,
                                  radius=np_random.uniform(0.01, 0.0375),
                                  mass=mass,
                                  static=static,
                                  collision=collision,
                                  friction=friction,
                                  visual=visual,
                                  gui_only=gui_only,
                                  color=color)
        else:
            print(f"Error: '{use_specific_primitive}'' in not a supported primitive. "
                  "Pleasure use 'box', 'cylinder' or 'sphere.")
