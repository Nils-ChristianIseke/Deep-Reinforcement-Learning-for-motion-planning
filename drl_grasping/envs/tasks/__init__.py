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


from gym import logger as gym_logger
from gym_ignition.utils import logger as gym_ign_logger
from gym.envs.registration import register
from ament_index_python.packages import get_package_share_directory
from os import path, environ

from .reach import Reach, ReachColorImage, ReachDepthImage, ReachOctree, ReachWithObstacles
from .grasp import Grasp, GraspOctree
from .inversekinematics import InverseKinematics,InverseKinematicsWithObstacles, InverseKinematicsWithMovingObstacles,InverseKinematicsWithManyMovingObstacles
# Set debug level
debug_level = environ.get('DRL_GRASPING_DEBUG_LEVEL', default='ERROR')
gym_ign_logger.set_level(
    level=getattr(gym_logger, debug_level),
    scenario_level=getattr(gym_logger, debug_level)
)

# Get path for worlds
worlds_dir = path.join(get_package_share_directory('drl_grasping'), 'worlds')

# Reach
REACH_MAX_EPISODE_STEPS: int = 100
REACH_AGENT_RATE: float = 2.5
REACH_PHYSICS_RATE: float = 100.0
REACH_RTF: float = 15.0
register(
    id='Reach-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': Reach,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            })
register(
    id='Reach-ColorImage-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': ReachColorImage,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            })
register(
    id='Reach-DepthImage-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': ReachDepthImage,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            })
register(
    id='Reach-Octree-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': ReachOctree,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'octree_depth': 4,
            'octree_full_depth': 2,
            'octree_include_color': False,
            'octree_n_stacked': 2,
            'octree_max_size': 20000,
            'verbose': False,
            })
register(
    id='Reach-OctreeWithColor-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': ReachOctree,
            'agent_rate': REACH_AGENT_RATE,
            'physics_rate': REACH_PHYSICS_RATE,
            'real_time_factor': REACH_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'octree_depth': 4,
            'octree_full_depth': 2,
            'octree_include_color': True,
            'octree_n_stacked': 2,
            'octree_max_size': 35000,
            'verbose': False,
            })

# Grasp
GRASP_MAX_EPISODE_STEPS: int = 100
GRASP_AGENT_RATE: float = 2.5
GRASP_PHYSICS_RATE: float = 250.0
GRASP_RTF: float = 10.0
register(
    id='Grasp-Octree-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': GraspOctree,
            'agent_rate': GRASP_AGENT_RATE,
            'physics_rate': GRASP_PHYSICS_RATE,
            'real_time_factor': GRASP_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'gripper_dead_zone': 0.25,
            'full_3d_orientation': False,
            'sparse_reward': True,
            'normalize_reward': False,
            'required_reach_distance': 0.1,
            'required_lift_height': 0.125,
            'reach_dense_reward_multiplier': 5.0,
            'lift_dense_reward_multiplier': 10.0,
            'act_quick_reward': -0.005,
            'outside_workspace_reward': 0.0,
            'ground_collision_reward': -1.0,
            'n_ground_collisions_till_termination': GRASP_MAX_EPISODE_STEPS,
            'curriculum_enable_workspace_scale': False,
            'curriculum_min_workspace_scale': 0.1,
            'curriculum_enable_object_count_increase': False,
            'curriculum_max_object_count': 4,
            'curriculum_enable_stages': False,
            'curriculum_stage_reward_multiplier': 7.0,
            'curriculum_stage_increase_rewards': True,
            'curriculum_success_rate_threshold': 0.75,
            'curriculum_success_rate_rolling_average_n': 100,
            'curriculum_restart_every_n_steps': 0,
            'curriculum_skip_reach_stage': False,
            'curriculum_skip_grasp_stage': True,
            'curriculum_restart_exploration_at_start': False,
            'max_episode_length': GRASP_MAX_EPISODE_STEPS,
            'octree_depth': 4,
            'octree_full_depth': 2,
            'octree_include_color': False,
            'octree_n_stacked': 3,
            'octree_max_size': 50000,
            'proprieceptive_observations': True,
            'verbose': False})
register(
    id='Grasp-OctreeWithColor-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=GRASP_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': GraspOctree,
            'agent_rate': GRASP_AGENT_RATE,
            'physics_rate': GRASP_PHYSICS_RATE,
            'real_time_factor': GRASP_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'gripper_dead_zone': 0.0,
            'full_3d_orientation': False,
            'sparse_reward': True,
            'normalize_reward': False,
            'required_reach_distance': 0.1,
            'required_lift_height': 0.125,
            'reach_dense_reward_multiplier': 5.0,
            'lift_dense_reward_multiplier': 10.0,
            'act_quick_reward': -0.005,
            'outside_workspace_reward': 0.0,
            'ground_collision_reward': -1.0,
            'n_ground_collisions_till_termination': GRASP_MAX_EPISODE_STEPS,
            'curriculum_enable_workspace_scale': False,
            'curriculum_min_workspace_scale': 0.1,
            'curriculum_enable_object_count_increase': False,
            'curriculum_max_object_count': 4,
            'curriculum_enable_stages': False,
            'curriculum_stage_reward_multiplier': 7.0,
            'curriculum_stage_increase_rewards': True,
            'curriculum_success_rate_threshold': 0.6,
            'curriculum_success_rate_rolling_average_n': 100,
            'curriculum_restart_every_n_steps': 0,
            'curriculum_skip_reach_stage': False,
            'curriculum_skip_grasp_stage': True,
            'curriculum_restart_exploration_at_start': False,
            'max_episode_length': GRASP_MAX_EPISODE_STEPS,
            'octree_depth': 4,
            'octree_full_depth': 2,
            'octree_include_color': True,
            'octree_n_stacked': 3,
            'octree_max_size': 75000,
            'proprieceptive_observations': True,
            'verbose': False})


IK_MAX_EPISODE_STEPS: int = 100
IK_AGENT_RATE: float = 2.5
IK_PHYSICS_RATE: float = 100.0
IK_RTF: float = 100.0
register(
    id='IK-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematics,
            'agent_rate': IK_AGENT_RATE,
            'physics_rate': IK_PHYSICS_RATE,
            'real_time_factor': IK_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            })



IK_WO_MAX_EPISODE_STEPS: int = 100
IK_WO_AGENT_RATE: float = 2.5
IK_WO_PHYSICS_RATE: float = 250.0
IK_WO_RTF: float = 100.0
register(
    id='IK-WO-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematicsWithObstacles,
            'agent_rate': IK_WO_AGENT_RATE,
            'physics_rate': IK_WO_PHYSICS_RATE,
            'real_time_factor': IK_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            'ground_collision_reward': -1.0,
            'obstacle_collision_reward': -1.0,
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })

IK_WO_MAX_EPISODE_STEPS: int = 100
IK_WO_AGENT_RATE: float = 2.5
IK_WO_PHYSICS_RATE: float = 250.0
IK_WO_RTF: float = 100.0
register(
    id='IK-WO-Gazebo-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematicsWithObstacles,
            'agent_rate': IK_WO_AGENT_RATE,
            'physics_rate': IK_WO_PHYSICS_RATE,
            'real_time_factor': IK_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            'ground_collision_reward': -1.0,
            'obstacle_collision_reward': -1.0,
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })

REACH_WO_MAX_EPISODE_STEPS: int = 10
REACH_WO_AGENT_RATE: float = 2.5
REACH_WO_PHYSICS_RATE: float = 100
REACH_WO_RTF: float = 10000
required_accuracy= 0.05
register(
    id='REACH-WO-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=REACH_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': ReachWithObstacles,
            'agent_rate': REACH_WO_AGENT_RATE,
            'physics_rate': REACH_WO_PHYSICS_RATE,
            'real_time_factor': REACH_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': 0,
            'required_accuracy': required_accuracy,
            'verbose': False,
            'ground_collision_reward': -1/(required_accuracy+0.00001),
            'obstacle_collision_reward': -1/(required_accuracy+0.00001),
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })


IK_WO_MAX_EPISODE_STEPS: int = 100
IK_WO_AGENT_RATE: float = 2.5
IK_WO_PHYSICS_RATE: float = 250.0
IK_WO_RTF: float = 100.0
register(
    id='IK-WO-Gazebo-v2',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematicsWithObstacles,
            'agent_rate': IK_WO_AGENT_RATE,
            'physics_rate': IK_WO_PHYSICS_RATE,
            'real_time_factor': IK_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            'ground_collision_reward': -1.0,
            'obstacle_collision_reward': -1.0,
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })


IK_WO_MAX_EPISODE_STEPS: int = 100
IK_WO_AGENT_RATE: float = 2.5
IK_WO_PHYSICS_RATE: float = 250.0
IK_WO_RTF: float = 1.0
register(
    id='IK-WMO-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematicsWithMovingObstacles,
            'agent_rate': IK_WO_AGENT_RATE,
            'physics_rate': IK_WO_PHYSICS_RATE,
            'real_time_factor': IK_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            'ground_collision_reward': -1.0,
            'obstacle_collision_reward': -1.0,
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })
IK_WO_MAX_EPISODE_STEPS: int = 100
IK_WO_AGENT_RATE: float = 2.5
IK_WO_PHYSICS_RATE: float = 250.0
IK_WO_RTF: float = 1
register(
    id='IK-WMMO-Gazebo-v0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=IK_WO_MAX_EPISODE_STEPS,
    kwargs={'world': path.join(worlds_dir, 'default.sdf'),
            'task_cls': InverseKinematicsWithManyMovingObstacles,
            'agent_rate': IK_WO_AGENT_RATE,
            'physics_rate': IK_WO_PHYSICS_RATE,
            'real_time_factor': IK_WO_RTF,
            'robot_model': 'panda',
            'restrict_position_goal_to_workspace': True,
            'sparse_reward': False,
            'act_quick_reward': -0.01,
            'required_accuracy': 0.05,
            'verbose': False,
            'ground_collision_reward': -1.0,
            'obstacle_collision_reward': -1.0,
            'n_ground_collisions_till_termination': 1,
            'n_obstacle_collisions_till_termination': 1,
            })
