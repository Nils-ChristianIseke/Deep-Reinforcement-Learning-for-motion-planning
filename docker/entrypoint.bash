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

## ROS 2
source /opt/ros/${ROS2_DISTRO}/setup.bash

## ROS 2 <-> IGN
source ${DRL_GRASPING_DIR}/ros_ign/install/local_setup.bash

## Main repository
source ${DRL_GRASPING_DIR}/drl_grasping/install/local_setup.bash

## O-CNN Octree
export PATH=${DRL_GRASPING_DIR}/O-CNN/octree/build:${PATH}
export PYTHONPATH=${DRL_GRASPING_DIR}/O-CNN/octree/build/python:${PYTHONPATH}

## Path to PBR textures
if [ -d "${DRL_GRASPING_DIR}/pbr_textures" ]; then
    # Use external textures (mounted as volume)
    export DRL_GRASPING_PBR_TEXTURES_DIR="${DRL_GRASPING_DIR}/pbr_textures"
else
    # Use the default textures
    export DRL_GRASPING_PBR_TEXTURES_DIR="${DRL_GRASPING_DIR}/default_pbr_textures"
fi

## Aliases for ease of configuration
alias _drl_grasping='cd ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping'
alias cfg_tasks='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/drl_grasping/envs/tasks/__init__.py'
alias cfg_hyperparams_td3='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/hyperparams/td3.yml'
alias cfg_hyperparams_sac='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/hyperparams/sac.yml'
alias cfg_hyperparams_tqc='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/hyperparams/tqc.yml'
alias cfg_ex_train='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/examples/ex_train.bash'
alias cfg_ex_enjoy='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/examples/ex_enjoy.bash'
alias cfg_ex_optimize='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/examples/ex_optimize.bash'
alias cfg_ex_enjoy_pretrained_agent='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/examples/ex_enjoy_pretrained_agent.bash'
alias cfg_ex_preload_replay_buffer='nano ${DRL_GRASPING_DIR}/drl_grasping/src/drl_grasping/examples/ex_preload_replay_buffer.bash'

## Appending source command to ~/.bashrc enables autocompletion (ENTRYPOINT alone does not support that)
grep -qxF '. "${DRL_GRASPING_DIR}/entrypoint.bash"' ${HOME}/.bashrc || echo '. "${DRL_GRASPING_DIR}/entrypoint.bash"' >>${HOME}/.bashrc
