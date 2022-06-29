# Deep Reinforcement Learning for motion planning
This repository is an extesion of: [drl_grasping](https://github.com/AndrejOrsula/drl_grasping). Please read through the [README](https://github.com/AndrejOrsula/drl_grasping) of the original repo first! This is necessary to understand this README.
We have deliberately chosen to focus this README only on the changes that we implemented.


In our Extension of the fantastic work of [AndrejOrsula](https://github.com/AndrejOrsula) we added few new Environments:
<details><summary>Newly added Environments (click to expand)</summary>
  
  1. InverseKinematics
  2. InverseKinematicsWithObstacles
  3. InverseKinematicsWithMovingObstacles
  4. InverseKinematicsWithManyMovingObstacles
  5. ReachWithObstacles
  
For a detailed explination of each task, go down to the environment section.
The naming of the environments 3 and 4 is a bit misleading. MovingObstacle refers to the fact that the obstacles are randomly spawend at the beginning of each episode, but they are staying at the same position during the whole episode.
  
</details>
The following animations are showing some results using the Panda robotic arm.
<p align="left" float="middle">
  <img width="50.0%" src="https://github.com/Nils-ChristianIseke/deepRLIK/blob/main/gifs/IK_TQC_100000.webp" alt="Evaluation of a trained policy on the InverseKinematic Task"/>
  <img width="40.0%" src="https://github.com/Nils-ChristianIseke/deepRLIK/blob/main/gifs/IK_WO_TQC_400000.webp" alt="Evaluation of a trained policy on the InverseKinematicTaskWithRandomObstacles"/>
</p>

Disclaimer: These instruction are based on the [original Repository](https://github.com/AndrejOrsula) and were adjusted to the Extension we are providing.

## Instructions

### Requirements
Take a look at the requirement section of the original [repository](https://github.com/AndrejOrsula/drl_grasping)

### Dependencies
Take a look at the the depenencies section of the original repository [repository](https://github.com/AndrejOrsula/drl_grasping)


### Docker Instructions
1. Pull this repository.
```git pull <link_to_repository>```
2. Get the docker image:
3. docker pull slin25/rl_motion_planning

#### VS CODE Remote Containers
One convinient way to edit the code e.g.: changing the reward function, or adding new tasks, is by connecting VS-Code to the container:
  1. Install [VS Code](https://code.visualstudio.com/download)
  2. Install the VS Code Extension [Remote Containers](https://code.visualstudio.com/docs/remote/containers)

  Now you can start developing inside the container by:
  1. Starting the container: 
    
  ```bash
      cd drl_grasping dir/docker
      sudo ./run.bash slin25/rl_motion_planning /bin/bash
   ```

  2. Connecting to the container as described [here](https://code.visualstudio.com/docs/remote/containers)
  3. inside the condatiner cd to the ros package dlr_grasping:
    
  ```bash
      cd /root/drl_grasping/drl_grasping/src/drl_grasping
   ```
   Now you are at the root of the ROS-Package.
   


### Training of Agent

Take a look at the the Training of Agent section of the original repository [repository](https://github.com/AndrejOrsula/drl_grasping)

If you want to see what is going on , you need to uncomment 'model.env.render("human")' in train.py (deepRLIK/scripts/train.py). To force the start of the simulation.

#### Continue Training on pretrained agents
We are also providing some pretrained agent, those can be found in the training directories. If you want to use them you need to change the
TRAINED_AGENT varibale in ex_train.bash. it shall point to one .zip file.

## Environments

Take a look at the the Enviroment section of the original repository [repository](https://github.com/AndrejOrsula/drl_grasping)

We added the following enviroments, to the original repository:
  
  <details><summary>New Environments (Work of this project) (click to expand)</summary>

  - [InverseKinematics](drl_grasping/envs/tasks/inverse_kinematics/inverse_kinematics.py) task
    Description: The agents goal is to calculate the necessary joint angles of the robotic arm to reach a random goal point -->
    The agent shall learn the inverse kinematic model of the arm.
    Environment: The environment contains the robotic arm, and a randomly spawned goal point.
    Observation: Position of the goal point and the endeffector of the robotic arm
    Action: The joint angles of the robotic arm.
    
    - [InverseKinematicsWithObstacles](drl_grasping/envs/tasks/inverse_kinematics_with_obstacles.py)
      -Description: The agents goal is to calculate the necessary joint angles of the robotic arm to reach a random goal point, while avoiding collisions with an obstacle
    Environment: The environment contains the robotic arm, a randomly spawned goal point and an obstacle.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacle
    Action: The joint angles of the robotic arm.
  - [InverseKinematicsWithMovingObstacles](drl_grasping/envs/tasks/inverse_kinematics_with_obstacles.py)
      -Description: The agents goal is to calculate the necessary joint angles of the robotic arm to reach a random goal point, while avoiding collisions with an obstacle
    Environment: The environment contains the robotic arm, a randomly spawned goal point and an obstacle.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacle
    Action: The joint angles of the robotic arm.
   [InverseKinematicsWithManyMovingObstacles](drl_grasping/envs/tasks/inverse_kinematics_with_obstacles.py)
      -Description: The agents goal is to calculate the necessary joint angles of the robotic arm to reach a random goal point, while avoiding collisions with an obstacle
    Environment: The environment contains the robotic arm, a randomly spawned goal point as well as a number of obstacles.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacles
    Action: The joint angles of the robotic arm.
  - [Reach](drl_grasping/envs/tasks/reach) task (extension of the orginal Reach Task)
      - [ReachWithObstacles](drl_grasping/envs/tasks/reach/reach.py)
    -Description: The agents goal is to calculate the necessary goal positions to move the robotic arm to reach a random goal point, while avoiding collisions with an obstacle. The inverse kinematic is calculated via MOVEIT!.
    Environment: The environment contains the robotic arm, a randomly spawned goal point and an obstacle.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacle
    Action: The goal point.  
  
  
  
  Inside the definition of each class some variables can be set, e.g.: For the InverseKinematicsWithMovingObstacles task. Especially important are the object, and obstacle related variables. For the newly implemented tasks the object and obstacle related variables (e.g.:_object_enable, _object_type, _object_dimension_volume, obstacle_type, etc.) define the properties of the goal point and the obstacle (where it is spawned, what it looks like etc.).  The standarts values are restricting the possible spawning volume of object and obstalce to a small volume. Thus keeping the observation space small. (Faster training). For a more general solution the spawning volume of both should be the same size as the workspace of the robot. 
</details>


<details><summary>Adding new environments and training the agent (click to expand)</summary>
To implement a new task / environment, the following steps are necessary:
  
  
  1. In the dir `/envs/task` add your task(e.g.: inversekinematics.py inside the inversekinematics dir)
  2. Register your task as gym environment inside `/envs/tasks/__init__.py`(e.g.: adding register(
    id='IK-Gazebo-v0',...kwargs={...,'task_cls': InverseKinematics,...)
  4. Add the hyperparams for your task `/hyperparams` (e.g. add IK-Gazebo-v0 with arguments to the tqc.yml)
  5. Adjust the arguments of `examples/ex_train.bash` (e.g. change ENV_ID to "IK-Gazebo-v0" and ALGO to "tqc")
  6. Uncommend model.env.render("human") in  `/scripts/train.py` if you want to see the simulation of the environment.
  7. Start the training by executing: `ros2 run drl_grasping ex_train.bash` in the running container
</details>


## Future Work
 From the author's point future work could focus on:
  
  - enlarging the spawning volume of obstacle and goal point to the whole workspace
  - Adding moving obstacles and goal_points
  - Adding obstacles of complex shape
  - Comparing the RL-Learning Approach for path planning with classic approaches of path planning
  - Making the task more complex by sensing the obstacle space via a camera (as it's done in the grasp task), instead of getting the positions of the obstacles via the gazebo API
  - Autotuning Hyperparameters
