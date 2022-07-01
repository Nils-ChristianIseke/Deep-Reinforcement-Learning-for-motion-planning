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


from gym_ignition.scenario import model_wrapper
from gym_ignition.utils import misc
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List


class Plane(model_wrapper.ModelWrapper):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'plane',
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 size: List[float] = (1.0, 1.0),
                 direction: List[float] = (0.0, 0.0, 1.0),
                 collision: bool = True,
                 friction: float = 1.0,
                 visual: bool = True):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Create SDF string for the model
        sdf = \
            f'''<sdf version="1.7">
            <model name="{model_name}">
                <static>true</static>
                <link name="{model_name}_link">
                    {
                    f"""
                    <collision name="{model_name}_collision">
                        <geometry>
                            <plane>
                                <normal>{direction[0]} {direction[1]} {direction[2]}</normal>
                                <size>{size[0]} {size[1]}</size>
                            </plane>
                        </geometry>
                        <surface>
                            <friction>
                                <ode>
                                    <mu>{friction}</mu>
                                    <mu2>{friction}</mu2>
                                    <fdir1>0 0 0</fdir1>
                                    <slip1>0.0</slip1>
                                    <slip2>0.0</slip2>
                                </ode>
                            </friction>
                        </surface>
                    </collision>
                    """ if collision else ""
                    }
                    {
                    f"""
                    <visual name="{model_name}_visual">
                        <geometry>
                            <plane>
                                <normal>{direction[0]} {direction[1]} {direction[2]}</normal>
                                <size>{size[0]} {size[1]}</size>
                            </plane>
                        </geometry>
                        <material>
                            <ambient>0.8 0.8 0.8 1</ambient>
                            <diffuse>0.8 0.8 0.8 1</diffuse>
                            <specular>0.8 0.8 0.8 1</specular>
                        </material>
                    </visual>
                    """ if visual else ""
                    }
                </link>
            </model>
        </sdf>'''

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
