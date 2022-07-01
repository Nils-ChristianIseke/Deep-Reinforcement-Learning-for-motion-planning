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


from drl_grasping.utils.model_collection_randomizer import ModelCollectionRandomizer
from gym_ignition.scenario import model_wrapper
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List


class RandomObject(model_wrapper.ModelWrapper):

    def __init__(self,
                 world: scenario.World,
                 name: str = 'object',
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_paths: str = None,
                 owner: str = 'GoogleResearch',
                 collection: str = 'Google Scanned Objects',
                 server: str = 'https://fuel.ignitionrobotics.org',
                 server_version: str = '1.0',
                 unique_cache: bool = False,
                 reset_collection: bool = False,
                 np_random=None):

        # Get a unique model name
        model_name = get_unique_model_name(world, name)

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        model_collection_randomizer = ModelCollectionRandomizer(model_paths=model_paths,
                                                                owner=owner,
                                                                collection=collection,
                                                                server=server,
                                                                server_version=server_version,
                                                                unique_cache=unique_cache,
                                                                reset_collection=reset_collection,
                                                                np_random=np_random)

        # Note: using default arguments here
        modified_sdf_file = model_collection_randomizer.random_model()

        # Insert the model
        ok_model = world.to_gazebo().insert_model(modified_sdf_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError('Failed to insert ' + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
