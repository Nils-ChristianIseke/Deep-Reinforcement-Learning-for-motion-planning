from drl_grasping.envs.tasks.manipulation import Manipulation
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import List, Tuple
import abc
import gym
import numpy as np
from drl_grasping.envs.models.robots import Panda

class InverseKinematics(Manipulation, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    
    # Set Collision Properties of Robot
    _robot_arm_collision: bool =True
    _robot_hand_collision: bool = True    
    
    # Add an Object as Target
    _object_enable: bool = True
    _object_type: str = 'sphere'
    _object_dimensions: List[float] = [0.05, 0.05, 0.05]
    _object_collision: bool = False
    _object_visual: bool = True
    _object_static: bool = True
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.95)
    # The object is spanwed inside a given Boundingbox defined by it's global center and it's dimensions
    _object_spawn_centre: Tuple[float, float, float] = \
        (0.4,
         0,
         0.3)
    _object_spawn_volume: Tuple[float, float, float] = \
        (0.3,
         0.3,
         0.1)
    
    _obstacle_enable: bool =False
    _ground_enable:bool = False
   
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
        Manipulation.__init__(self,
                              agent_rate=agent_rate,
                              robot_model=robot_model,
                              restrict_position_goal_to_workspace=restrict_position_goal_to_workspace,
                              verbose=verbose,
                              **kwargs)

        # Additional parameters
        self._sparse_reward: bool = sparse_reward
        self._act_quick_reward = act_quick_reward if act_quick_reward >= 0.0 else -act_quick_reward
        self._required_accuracy: float = required_accuracy

        # Flag indicating if the task is done (performance - get_reward + is_done)
        self._is_done: bool = False

        # Distance to target in the previous step (or after reset)
        self._previous_distance: float = None

    def create_action_space(self) -> ActionSpace:

        # 0:7 Joint_angles
        
        joint_limits = Panda.get_joint_limits()
        joint_limits_lower = np.array([limit[0] for limit in joint_limits[:-2]])
        joint_limits_upper = np.array([limit[1] for limit in joint_limits[:-2]])
        
        print(joint_limits)
        
        print(joint_limits_lower)
        # FIX IF TIME: 
        # If individual joint_limits are specified following error is thrown --> Investigate if there is time
        # file "/usr/local/lib/python3.8/dist-packages/gym/spaces/box.py", line 69, in __init__
        # low_precision = _get_precision(self.low.dtype)
        # AttributeError: 'list' object has no attribute 'dtype
    
        return gym.spaces.Box(low=joint_limits_lower,
                              high=joint_limits_upper,
                              dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:

        # 0:3 - (x, y, z) end effector position
        # 3:6 - (x, y, z) target position
        # Note: These could theoretically be restricted to the workspace and object spawn area instead of inf
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                                shape=(6,),
                              dtype=np.float32)

    def set_action(self, action: Action):
        if self._verbose:
            print(f"action: {action}")
        
        # Set joint_angles
        self.set_jointangles(action)

        # Plan and execute motion to joint_angles
        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

    def get_observation(self) -> Observation:

        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()

        # Create the observation
        observation = Observation(np.concatenate([ee_position,
                                                  target_position]))

        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation

    def get_reward(self) -> Reward:

        reward = 0.0

        # Compute the current distance to the target
        current_distance = self.get_distance_to_target()

        # Mark the episode done if target is reached
        if current_distance < self._required_accuracy:
            self._is_done = True
            reward += 1/(current_distance)**2
            # if self._sparse_reward:
            #     reward += 1.0

        # Give reward based on how much closer robot got relative to the target for dense reward
        if not self._sparse_reward:
            reward += min(1/(current_distance)**2,1/(self._required_accuracy)**2)
            
            self._previous_distance = current_distance

        # Subtract a small reward each step to provide incentive to act quickly (if enabled)
        reward -= self._act_quick_reward
        print(current_distance)
        print(reward)
        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:

        done = self._is_done

        if self._verbose:
            print(f"done: {done}")

        return done

    def reset_task(self):

        self._is_done = False

        # Compute and store the distance after reset if using dense reward
        if not self._sparse_reward:
            self._previous_distance = self.get_distance_to_target()

        if self._verbose:
            print(f"\ntask reset")

    def get_distance_to_target(self) -> Tuple[float, float, float]:

        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()

        # Compute the current distance to the target
        return np.linalg.norm([ee_position[0] - target_position[0],
                               ee_position[1] - target_position[1],
                               ee_position[2] - target_position[2]])

    def get_target_position(self) -> Tuple[float, float, float]:

        target_object = self.world.get_model(self.object_names[0]).to_gazebo()
        return target_object.get_link(link_name=target_object.link_names()[0]).position()
