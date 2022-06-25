from drl_grasping.envs.tasks.manipulation import Manipulation
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import List, Tuple
import abc
import gym
import numpy as np
from drl_grasping.envs.models.robots import Panda

class InverseKinematicsWithObstacles(Manipulation, abc.ABC):

    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    _robot_arm_collision: bool =True
    _robot_hand_collision: bool = True    
    
    

    _object_enable: bool = True
    _object_type: str = 'sphere'
       
    _object_dimensions: List[float] = [0.05]
    _object_collision: bool = False
    _object_visual: bool = True
    _object_static: bool = True
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.95)
    # With those positions sometimes the shortest trajectory is obstacle free.
    # _object_spawn_centre: Tuple[float, float, float] = \
    #     (0.6,
    #      0,
    #      0.1)
    # _object_spawn_volume: Tuple[float, float, float] = \
    #     (0.3,
    #      0.1,
    #      0.2)

    # With those positions the shortest trajectory is never obstacle free.
    _object_spawn_centre: Tuple[float, float, float] = \
        (0.6,
         0,
         0.05)
    _object_spawn_volume: Tuple[float, float, float] = \
        (0.3,
         0.1,
         0.1)
    
    _workspace_volume: Tuple[float, float, float] = _object_spawn_volume
    _workspace_centre: Tuple[float, float, float] = (
        _object_spawn_centre)

    _obstacle_enable: bool =True
    _obstacle_type: str = 'box'
     # With those dimensions the shortest trajectory is sometimes obstacle free.
    # _obstacle_dimensions: List[float] = [0.05, 0.1, 0.4]
         # With those dimensions the shortest trajectory is never obstacle free.
    _obstacle_dimensions: List[float] = [0.05, 0.1, 0.6]
    _obstacle_collision: bool = True
    _obstacle_visual: bool = True
    _obstacle_static: bool = True
    _obstacle_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    _obstacle_spawn_centre: Tuple[float, float, float] = \
        (0.4,
         0,
         _obstacle_dimensions[2]/2,)
    # _obstacle_spawn_volume_proportion: float = 0.75
    # _obstacle_spawn_volume: Tuple[float, float, float] = \
    #     (_obstacle_spawn_volume_proportion*_workspace_volume[0],
    #      _obstacle_spawn_volume_proportion*_workspace_volume[1],
    #      _obstacle_spawn_volume_proportion*_workspace_volume[2])
    _obstacle_quat_xyzw= (0,0,1,0)
    _obstacle_mass=1
                                    
                
    
    
    
    _ground_enable:bool = True
    _ground_position: Tuple[float, float, float] = (0, 0, 0)
    _ground_quat_xyzw: Tuple[float, float, float, float] = (0, 0, 0, 1)
    _ground_size: Tuple[float, float] = (1.25, 1.25)
    def __init__(self,
                 agent_rate: float,
                 robot_model: str,
                 restrict_position_goal_to_workspace: bool,
                 sparse_reward: bool,
                 act_quick_reward: float,
                 required_accuracy: float,
                 verbose: bool,
                 ground_collision_reward: float,
                 obstacle_collision_reward:float,
                 n_ground_collisions_till_termination:int,
                 n_obstacle_collisions_till_termination:int,
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
        # Flag indication if the tasked failed due to many collisions with ground or obstacle
        self._is_failure: bool = False
        # Distance to target in the previous step (or after reset)
        self._previous_distance: float = None
        
      
        self._ground_collision_reward = ground_collision_reward \
            if ground_collision_reward >= 0.0 else -ground_collision_reward
        self._obstacle_collision_reward = ground_collision_reward \
            if obstacle_collision_reward >= 0.0 else -obstacle_collision_reward
        self._ground_collision_counter: int = 0
        self._obstacle_collision_counter: int = 0
        self._n_ground_collisions_till_termination = n_ground_collisions_till_termination
        self._n_obstacle_collisions_till_termination = n_obstacle_collisions_till_termination
    def create_action_space(self) -> ActionSpace:

        # 0:3 - (x, y, z) displacement
        #     - rescaled to metric units before use
        
        joint_limits = Panda.get_joint_limits()
        joint_limits_lower = [limit[0] for limit in joint_limits]
        joint_limits_upper = [limit[1] for limit in joint_limits]
        
        # If individual joint_limits are specified following error is thrown --> Investigate if there is time
        # file "/usr/local/lib/python3.8/dist-packages/gym/spaces/box.py", line 69, in __init__
        # low_precision = _get_precision(self.low.dtype)
        # AttributeError: 'list' object has no attribute 'dtype
        return gym.spaces.Box(low=min(joint_limits_lower),
                              high=max(joint_limits_upper),
                              shape=(7,),
                              dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:
        # Observation space
        # 0:2 - (x, y, z) end effector position
        # 3:5 - (x, y, z) target position
        # 6:8 - (x, y, z) obstacle position
        # 8:11 - () obstacle dimensions
        # Note: These could theoretically be restricted to the workspace and object spawn area instead of inf
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                                shape=(12,),
                              dtype=np.float32)

    def set_action(self, action: Action):
        if self._verbose:
            print(f"action: {action}")
        
        # Set joint_angles
        self.set_jointangles([float(joint_angle)for joint_angle in action])

        # Plan and execute motion to joint_angles
        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

    def get_observation(self) -> Observation:

        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()
        obstacle_position = self.get_obstacle_position()
        obstacle_dimensions = self._obstacle_dimensions
        # obstacle_orientation = self.get_obstacle_orientation()
        # Create the observation
        observation = Observation(np.concatenate([ee_position,
                                                  target_position,
                                                  obstacle_position,
                                                  obstacle_dimensions]))
        if self._verbose:
            print(f"\nobservation: {observation}")

        # Return the observation
        return observation

    # def get_reward(self) -> Reward:

    #     reward = 0.0

    #     # Compute the current distance to the target
    #     current_distance = self.get_distance_to_target()

    #     # Mark the episode done if target is reached
    #     if current_distance < self._required_accuracy:
    #         self._is_done = True
    #         if self._sparse_reward:
    #             reward += 1.0

    #     # Give reward based on how much closer robot got relative to the target for dense reward
    #     if not self._sparse_reward:
    #         reward += self._previous_distance - current_distance
    #         self._previous_distance = current_distance
        
        
    #     neg_reward = self._get_reward_ALL()
    #     # Normalize the negative reward if desired
    #     # neg_reward *= self._normalize_negative_reward_multiplier

    #     # Sum all positive rewards with the negative reward
    #     reward += neg_reward
    #     if self._verbose:
    #         print(f"reward: {reward}")

    #     return Reward(reward)
#  def get_reward(self) -> Reward:

#         reward = 0.0

#         # Compute the current distance to the target
#         current_distance = self.get_distance_to_target()

#         # Mark the episode done if target is reached
#         if current_distance < self._required_accuracy:
#             self._is_done = True
#             reward += 1/(self._required_accuracy+0.0001)
#             if self._sparse_reward:
#                 reward += 1.0
#         else:
#             reward += min(1/(self._required_accuracy+0.0001),1/(current_distance+0.0001))

#         # Give reward based on how much closer robot got relative to the target for dense reward
#         if not self._sparse_reward:
#             reward += self._previous_distance - current_distance
#             self._previous_distance = current_distance
        
        
#         neg_reward = self._get_reward_ALL()
#         # Normalize the negative reward if desired
#         # neg_reward *= self._normalize_negative_reward_multiplier

#         # Sum all positive rewards with the negative reward
#         reward += neg_reward
#         if self._verbose:
#             print(f"reward: {reward}")

#         return Reward(reward)

    def get_reward(self) -> Reward:

        reward = 0.0

        # Compute the current distance to the target
        current_distance = self.get_distance_to_target()

        # Mark the episode done if target is reached
        if current_distance < self._required_accuracy:
            self._is_done = True
            
            # if self._sparse_reward:
            reward += 1.0

        # Give reward based on how much closer robot got relative to the target for dense reward
        if not self._sparse_reward:
            reward += self._previous_distance - current_distance
            self._previous_distance = current_distance

        # Subtract a small reward each step to provide incentive to act quickly (if enabled)
        reward -= self._act_quick_reward
        reward += self._get_reward_ALL()
        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)


    def _get_reward_ALL(self) -> float:

        # Subtract a small reward each step to provide incentive to act quickly
        reward = -self._act_quick_reward

        # Return reward of -1.0 if robot collides with the ground plane (and terminate when desired)
        if self.check_ground_collision():
            reward -= self._ground_collision_reward
            self._ground_collision_counter += 1
            self._is_failure = self._ground_collision_counter >= self._n_ground_collisions_till_termination
            
            if self._verbose:
                print("Robot collided with the ground plane.")
        if self.check_obstacle_collision():
            reward -= self._obstacle_collision_reward
            self._obstacle_collision_counter +=1
            self._is_failure = self._obstacle_collision_counter >= self._n_obstacle_collisions_till_termination
            
            if self._verbose:
                print("Robot collided with an obstacle.")
        return reward







    def is_done(self) -> bool:

        if self._is_done:
            print("Success")
            return True
        if self._is_failure:
             print("Failed")
             return True

    def reset_task(self):

        self._is_done = False
        self._is_failure = False
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

        target_object =     self.world.get_model(self.object_names[0]).to_gazebo()
        return target_object.get_link(link_name=target_object.link_names()[0]).position()
    def get_obstacle_position(self) -> Tuple[float, float, float,float, float, float]:
        target_object =     self.world.get_model(self.obstacle_names[0]).to_gazebo()
        return target_object.get_link(link_name=target_object.link_names()[0]).position()
    # def get_obstacle_orientation(self) -> Tuple[float, float, float,float, float, float]:
    #     target_object =     self.world.get_model(self.obstacle_names[0]).to_gazebo()
    #     return target_object.get_link(link_name=target_object.link_names()[0]).orientation()
    # def get_obstacle_dimensions(self) -> Tuple[float, float, float,float, float, float]:
    #     target_object =     self.world.get_model(self.obstacle_names[0]).to_gazebo()
    #     print(type(target_object))
    #     print(type(target_object.get_link(link_name=target_object.link_names()[0])))
    #     return target_object.get_shape_bounding_box()
    def check_ground_collision(self) -> bool:
        """
        Returns true if robot links are in collision with the ground.
        """
        ground = self.world.get_model(self.ground_name)
        
        for contact in ground.contacts():
            for obstacle_name in self.obstacle_names:
                if self.robot_name in contact.body_b and not self.robot_base_link_name in contact.body_b and not obstacle_name in contact.body_b:
                    return True
        return False

    def check_obstacle_collision(self) -> bool:
        for obstacle_name in self.obstacle_names:
            obstacle = self.world.get_model(obstacle_name)
            for contact in obstacle.contacts():
                if self.robot_name in contact.body_b:
                    return True
        return False