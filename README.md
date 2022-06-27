# Deep Reinforcement Learning for inverse Kinematics
This repository is an extesion of: [drl_grasping](https://github.com/AndrejOrsula/drl_grasping). Please read through it's [README](https://github.com/AndrejOrsula/drl_grasping). 


In our Extension of the fantastic work of [AndrejOrsula](https://github.com/AndrejOrsula) we added new Environments:
<details><summary>Newly added Environments (click to expand)</summary>
  
  1. InverseKinematics
  2. InverseKinematicsWithObstacles
  3. ForwardKinematicsWithObstacles

</details>
The following animations are showing some results using the Panda robotic arm.
<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_panda.webp" alt="Evaluation of a trained policy on novel scenes for Panda robot"/>
</p>
<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/media/webp/sim_ur5_rg2.webp" alt="Evaluation of a trained policy on novel scenes for UR5 robot"/>
</p>

Disclaimer: These instruction are based on the [original Repository](https://github.com/AndrejOrsula) and were adjusted to the Extension we are providing.
We added some parts and deleted parts which are not relevant for our contribution.

## Instructions



### Requirements

- **GPU:** CUDA is required to process octree observations on GPU.
  - Everything else should function normally on CPU, i.e. environments with other observation types.
- VS Code
- Remote Containers
<details><summary> Developing with Docker (click to expand)</summary>

### Requirements

- **OS:** Any system that supports [Docker](https://docs.docker.com/get-docker) should work (Linux, Windows, macOS).
  - Only Ubuntu 22.04 was tested.
- **GPU:** CUDA is required to process octree observations on GPU. Therefore, only Docker images with CUDA support are currently available.

### Dependencies

Before starting, make sure your system has a setup for using [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker), e.g.:

```bash
# Docker
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
# Nvidia Docker
distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Pre-built Docker Image

The easiest way to try out this project is by using a pre-built Docker image that can be pulled from [Docker Hub](https://hub.docker.com/repository/docker/andrejorsula/drl_grasping). Currently, there is only a development image available that also contains the default testing datasets (huge, but it is easy to use and allows editing and recompiling). You can pull the `latest` tag with the following command. Alternatively, each release has also its associated tag, e.g. `1.0.0`.

```bash
docker pull andrejorsula/drl_grasping:latest
```

For running of the container, please use the included [docker/run.bash](docker/run.bash) script that is included with this repo. It significantly simplifies the setup with volumes and allows use of graphical interfaces for Ignition Gazebo GUI client and RViZ.

Open a Terminal and cd to the directory where the run.bash is located:
```bash
  cd drl_grasping dir/docker
  ```
Execute run.bash:
```bash
./run.bash andrejorsula/drl_grasping:latest /bin/bash
```

The easiest way to edit the code e.g.: changing the reward function, or adding new tasks, is by connecting VS-Code to the container:
  1. Install [VS Code](https://code.visualstudio.com/download)
  2. Install the VS Code Extension [Remote Containers](https://code.visualstudio.com/docs/remote/containers)

  Now you can start developing inside the container by:
  1. Starting a container (As described above: 
    
    ```bash
      cd drl_grasping dir/docker
    ```
    
    ```bash
      ./run.bash andrejorsula/drl_grasping:latest /bin/bash
    ```)
  2. Connecting to the container as described [here](https://code.visualstudio.com/docs/remote/containers)
  3. cd to dlr_grasping:
    
  ```bash
    cd /root/drl_grasping/drl_grasping/src/drl_grasping
   ```
  4. Start developing :) If you want another terminal e.g. for running tensorboard, open a new terminal and execute: 
    ```bash
      docker exec -ti container_id bash
    ```
    Where container_id is the id of the container, which is shown by executing:
    ```bash
      docker ps
    ```
  
  
<details><summary>Training New Agents (click to expand)</summary>


### Training of Agent

To train your own agent, you can start with the [`ex_train.bash`](examples/ex_train.bash) example. You can customise this example script,  configuration of the environment and all hyperparameters to your needs (see below). By default, headless mode is used during training to reduce computational load. If you want to see what is going on, you need to uncomment 'model.env.render("human")' in train.py (deepRLIK/scripts/train.py). 

```bash
ros2 run drl_grasping ex_train.bash
```

### Enjoying of Trained Agents

To enjoy an agent that you have trained yourself, look into [`ex_enjoy.bash`](examples/ex_enjoy.bash) example. Similar to training, change the environment ID, algorithm and robot model. Furthermore, select a specific checkpoint that you want to run. RViZ 2 and Ignition Gazebo GUI client are enabled by default.

```bash
ros2 run drl_grasping ex_enjoy.bash
```

</details>
  
</details>

## Environments

This repository contains environments for robotic manipulation that are compatible with [OpenAI Gym](https://github.com/openai/gym). All of these make use of [Ignition Gazebo](https://ignitionrobotics.org) robotic simulator, which is interfaced via [Gym-Ignition](https://github.com/robotology/gym-ignition).

Currently, the following environments are included inside this repository. Take a look at their [gym environment registration](drl_grasping/envs/tasks/__init__.py) and source code if you are interested in configuring them. There is a lot of parameters trying different RL approaches and techniques, so it is currently a bit messy (might get cleaned up if I have some free time for it).

<details><summary>Original Environments (click to expand)</summary>

  - [Grasp](drl_grasping/envs/tasks/grasp) task
    - Observation variants
      - [GraspOctree](drl_grasping/envs/tasks/grasp/grasp_octree.py), with and without color features
      - GraspColorImage (RGB image) and GraspRgbdImage (RGB-D image) are implemented on [image_obs](https://github.com/AndrejOrsula/drl_grasping/tree/image_obs) branch. However, their implementation is currently only for testing and comparative purposes.
    - Curriculum Learning: Task includes [GraspCurriculum](drl_grasping/envs/tasks/grasp/curriculum.py), which can be used to progressively increase difficulty of the task by automatically adjusting the following environment parameters based on the current success rate.
      - Workspace size
      - Number of objects
      - Termination state (task is divided into hierarchical sub-tasks with aim to further guide the agent).
      - This part does not bring any improvements based on experimental results, so do not bother using it.
    - Demonstrations: Task contains a simple scripted policy that can be applied to collect demonstrations, which can then be used to pre-load a replay buffer for training with off-policy RL algorithms.
      - It provides a slight increase for early learning, however, experiments indicate that it degrades the final success rate (probably due to introduction of bias early on). Therefore, do not use demonstrations if possible, at least not with this environment.
  - [Reach](drl_grasping/envs/tasks/reach) task (a simplistic environment for testing stuff)
    - Observation variants
      - [Reach](drl_grasping/envs/tasks/reach/reach.py) - simulation states
      - [ReachColorImage](drl_grasping/envs/tasks/reach/reach_color_image.py)
      - [ReachDepthImage](drl_grasping/envs/tasks/reach/reach_depth_image.py)
      - [ReachOctree](drl_grasping/envs/tasks/reach/reach_octree.py), with and without color features
  </details>
  
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
  - [InverseKinematicsWithRandomObstacles](drl_grasping/envs/tasks/inverse_kinematics_with_obstacles.py)
      -Description: The agents goal is to calculate the necessary joint angles of the robotic arm to reach a random goal point, while avoiding collisions with an obstacle
    Environment: The environment contains the robotic arm, a randomly spawned goal point and an obstacle.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacle
    Action: The joint angles of the robotic arm.
  - [Reach](drl_grasping/envs/tasks/reach) task (extension of the orginal Reach Task)
      - [ReachWithObstacles](drl_grasping/envs/tasks/reach/reach.py)
    -Description: The agents goal is to calculate the necessary goal positions to move the robotic arm to reach a random goal point, while avoiding collisions with an obstacle. The inverse kinematic is calculated via MOVEIT!.
    Environment: The environment contains the robotic arm, a randomly spawned goal point and an obstacle.
    Observation: Position of the goal point, the endeffector of the robotic arm and position + orientation of the obstacle
    Action: The goal point.  
  
  
  
  Inside the definition of each class some variables can be set, e.g.: For the InverseKinematicsWithRandomObstacles task. Especially important are the object, and obstacle related variables. For the newly implemented tasks the object and obstacle related variables (e.g.:_object_enable, _object_type, _object_dimension_volume, obstacle_type, etc.) define the properties of the goal point and the obstacle (where it is spawned, what it looks like etc.).  The standarts values are restricting the possible spawning volume of object and obstalce to a small volume, to keep the observation space for the RL-Agent small. For a more general solution the spawning volume of both should be the same size as the workspace of the robot. 
</details>


 <details><summary>Future Work (click to expand)</summary>
 From the author's point future work could focus on:
  
  - enlarging the spawning volume of obstacle and goal point
  - Adding more than 1 obstacle
  - Adding moving obstacles
  - Adding moving obstacles and goal_points
  - Adding obstacles of complex shape
  - Comparing the RL-Learing Approach for path planning with classic approaches to path planning
  - Making the task more complex by sensing the obstacle space via a camera (as it's done in the grasp task), instead of getting the positions of the obstacles via the gazebo API
</details>
<details><summary>Adding new environments (click to expand)</summary>
To implement a new task / environment, the following steps are necessary:
  
  1. In the dir `/envs/task` add your task(e.g.: inversekinematics.py inside the inversekinematics dir)
  2. Register your task as gym environment inside `/envs/tasks/__init__.py (e.g.: adding register(
    id='IK-Gazebo-v0',...kwargs={...,'task_cls': InverseKinematics,...)
  4. Add the hyperparams for your task `/hyperparams` (e.g. add IK-Gazebo-v0 with arguments to the tqc.yml)
  5. Adjust the arguments of `examples/ex_train.bash` (e.g. change ENV_ID to "IK-Gazebo-v0" and ALGO to "tqc")
  6. Uncommend model.env.render("human") in  `/scripts/train.py` if you want to see the simulation of the env.
</details>
<details><summary>Training New Agents (click to expand)</summary>

 ### Domain Randomization

These environments can be wrapped by a randomizer in order to introduce domain randomization and improve generalization of the trained policies, which is especially beneficial for Sim2Real transfer.

<p align="center" float="middle">
  <img width="100.0%" src="https://github.com/AndrejOrsula/master_thesis/raw/media/graphics/implementation/domain_randomisation.png" alt="Examples of domain randomization for the Grasp task"/>
</p>

The included [ManipulationGazeboEnvRandomizer](drl_grasping/envs/randomizers/manipulation.py) allows randomization of the following properties at each reset of the environment.

- Object model - primitive geometry
  - Random type (box, sphere and cylinder are currently supported)
  - Random color, scale, mass, friction
- Object model - mesh geometry
  - Random type (see [Object Model Database](#object-model-database)) 
  - Random scale, mass, friction
- Object pose
- Ground plane texture
- Initial robot configuration
- Camera pose

### Supported Robots

Only [Franka Emika Panda](https://github.com/AndrejOrsula/panda_ign) is supported.


## Reinforcement Learning

This project makes direct use of [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) as well as [sb3_contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib). Furthermore, scripts for training and evaluation are largely inspired by [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

### Hyperparameters

Hyperparameters for training of RL agents can be found in [hyperparams](hyperparams) directory. [Optuna](https://github.com/optuna/optuna) was used to autotune some of them, but certain algorithm/environment combinations require far more tuning (especially TD3). If needed, you can try running Optuna yourself, see [`ex_optimize`](examples/ex_optimize.bash) example.

## Directory Structure

```bash
├── drl_grasping        # Primary Python module of this project
    ├── algorithms      # Definitions of policies and slight modifications to RL algorithms
    ├── envs            # Environments for grasping (compatible with OpenAI Gym)
        ├── tasks       # Tasks for the agent that are identical for simulation
        ├── randomizers # Domain randomization of the tasks, which also populates the world
        └── models      # Functional models for the environment (Ignition Gazebo)
    ├── control         # Control for the agent
    ├── perception      # Perception for the agent
    └── utils           # Other utilities, used across the module
├── examples            # Examples for training and enjoying RL agents
├── hyperparams         # Hyperparameters for training RL agents
├── scripts             # Helpful scripts for training, evaluating, ... 
├── launch              # ROS 2 launch scripts that can be used to help with setup
├── docker              # Dockerfile for this project
└── drl_grasping.repos  # List of other dependencies created for `drl_grasping`
```

---

In case you have any problems or questions, feel free to open an [Issue](https://github.com/AndrejOrsula/drl_grasping/issues/new) or a [Discussion](https://github.com/AndrejOrsula/drl_grasping/discussions/new).
# deepRL_IK
# deepRL_IK
# deepRL_IK
# deepRL_IK
# deepRLIK
# deepRLIK
# deepRLIK
# deepRLIK
