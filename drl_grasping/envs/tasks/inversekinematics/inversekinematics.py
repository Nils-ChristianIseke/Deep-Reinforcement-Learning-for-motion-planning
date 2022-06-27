from drl_grasping.envs.tasks.manipulation import Manipulation
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from typing import List, Tuple
import abc
import gym
import numpy as np
from drl_grasping.envs.models.robots import Panda

class InverseKinematics(Manipulation, abc.ABC):
    # In this task the agent shall learn the inverse kinematic model of the robot. Thus, given a point in the workspace it should learn a policy to
    # calculate the needed joint angles to get there.
    # Observation: Position of the endeffector, position of the goal
    # Action: joint angles of the robotic arm
    # Reward: Positive reward for reaching the goal, or getting closer to it, negativ reward for colliding with an obstacle or the ground




    # Overwrite parameters for ManipulationGazeboEnvRandomizer
    
    # Set Collision Properties of Robot
    _robot_arm_collision: bool = True
    _robot_hand_collision: bool = True    
    
    # Add an Object as Target
    # This object has no collision properties. It's task is to define and visualize the goal point
    _object_enable: bool = True
    _object_type: str = 'sphere'
    _object_dimensions: List[float] = [0.05, 0.05, 0.05]
    _object_collision: bool = False
    _object_visual: bool = True
    _object_static: bool = True
    _object_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.95)

    # The object is spanwed inside a given Boundingbox defined by it's global center and it's dimensions.
    # By choosing the dimensions of the spawning volume, it is possible to change the size of the observation space. 
    _object_spawn_centre: Tuple[float, float, float] = \
        (0.4,
         0,
         0.3)
    _object_spawn_volume: Tuple[float, float, float] = \
        (0.3,
         0.3,
         0.1)

    _obstacle_random_pose_spawn = True                   
    _obstacle_random_poistion_spawn = True
    _obstacle_random_orientation_spawn = False
    # For this task we do not want obstacles or ground.
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
        """Create the action space for the robotic arm. The action space is defined by the upper and lower limits of the robotic arm which, are taken from it's Class definition.

        Returns:
            ActionSpace: gym.spaces.Box(), containing the lower and uppper joint_limits of the robotic arm
        """
        # 0:7 Joint_angles
        
        joint_limits = Panda.get_joint_limits()
        joint_limits_lower = np.array([limit[0] for limit in joint_limits[:-2]])
        joint_limits_upper = np.array([limit[1] for limit in joint_limits[:-2]])

        return gym.spaces.Box(low=joint_limits_lower,
                              high=joint_limits_upper,
                              dtype=np.float32)

    def create_observation_space(self) -> ObservationSpace:
        """Creates the Observation spaces, defines the limits in which observations are valid.
        Returns:
            ObservationSpace:  # gym.spaces.Box(), containing all possible values of observations 0:2 - (x, y, z) end effector position 3:5 - (x, y, z) target position
        """
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                                shape=(6,),
                              dtype=np.float32)

    def set_action(self, action: Action) -> None:
        """Defines how an action is created. Sets the joint_angles of the robotic arm to the value given by the reinforcement learning agent.
        The motion plan between the actual joint position and the goal joint position is calculated by moveit2.

        Args:
            action (Action): _description_
        """
        if self._verbose:
            print(f"action: {action}")
        
        # Set joint_angles
        self.set_jointangles([float(joint_angle)for joint_angle in action])

        # Plan and execute motion to joint_angles
        self.moveit2.plan_kinematic_path(allowed_planning_time=0.1)
        self.moveit2.execute()

    def get_observation(self) -> Observation:
        """Defines how the agent is getting information. In this case it is getting the position of the endeffector and the goal point from 
        the gazebo API.

        Returns:
            Observation: np.array: 0:2 -(x,y,z) end_effector position, 3:5 - (x,y,z) goal_position
        """
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
        """Calculating the reward. 
        Dense reward:
        A poisitve reward is assigned when getting closer to the target the in the previous step. A negative reward is assigned when the distance to the target
        increases. As soon as the robot collides with an obstacle a negative reward is assigned. For reaching the goal point, a positive reward is assigned.
        Sparse Reward:
        A reward is only assigned if the robot reaches the goal or collides with an obstacle.

        Returns:
            Reward: _description_
        """
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

        # Act quickly reward is assigned for rewarding a fast solution (steps) (if enabled)
        reward -= self._act_quick_reward
        if self._verbose:
            print(f"reward: {reward}")

        return Reward(reward)

    def is_done(self) -> bool:
        """Checks if the condition to end the episode are fullfilled.

        Returns:
            bool: True if the ending position is fullfilled. Falls otherwise.
        """
        done = self._is_done

        if self._verbose:
            print(f"done: {done}")

        return done

    def reset_task(self) -> None: 
        """Resets the variables _is_done and _previous_distance
        """
        self._is_done = False

        # Compute and store the distance after reset if using dense reward
        if not self._sparse_reward:
            self._previous_distance = self.get_distance_to_target()

        if self._verbose:
            print(f"\ntask reset")

    def get_distance_to_target(self) -> Tuple[float, float, float]:
        """Calculates the distance to the target.

        Returns:
            Tuple[float, float, float]: 1:3 -> (x,y,z) position of the endeffector
        """
        # Get current end-effector and target positions
        ee_position = self.get_ee_position()
        target_position = self.get_target_position()

        # Compute the current distance to the target
        return np.linalg.norm([ee_position[0] - target_position[0],
                               ee_position[1] - target_position[1],
                               ee_position[2] - target_position[2]])

    def get_target_position(self) -> Tuple[float, float, float]:
        """Gets the position of the target.

        Returns:
            Tuple[float, float, float]: 0:2 -> (x,y,z) position of the endeffector
        """
        target_object = self.world.get_model(self.object_names[0]).to_gazebo()
        return target_object.get_link(link_name=target_object.link_names()[0]).position()
