#!/usr/bin/env bash

## Random seed to use for both the environment and agent (-1 for random)
SEED="77"

## ID of the environment
## Reach
# ENV_ID="Reach-Gazebo-v0"
# ENV_ID="Reach-ColorImage-Gazebo-v0"
# ENV_ID="Reach-Octree-Gazebo-v0"
# ENV_ID="Reach-OctreeWithColor-Gazebo-v0"
## Grasp
# ENV_ID="Grasp-Octree-Gazebo-v0"
# ENV_ID="Grasp-OctreeWithColor-Gazebo-v0"
# ENV_ID="IK-WO-Gazebo-v0"
ENV_ID="IK-Gazebo-v0"

## Robot model
ROBOT_MODEL="panda"
# ROBOT_MODEL="ur5_rg2"
# ROBOT_MODEL="kinova_j2s7s300"

## Algorithm to use
# ALGO="sac"
# ALGO="td3"
ALGO="tqc"
# ALGO="a2c"
#ALGO="ppo"
## Arguments for the environment
ENV_ARGS="robot_model:\"${ROBOT_MODEL}\""

## Checkpoint to load
CHECKPOINT=490000

## Path the parent training directory
TRAINING_DIR="training"
## Path to logs
LOG_DIR=""${TRAINING_DIR}"/"${ENV_ID}"/logs"
## Path to reward logs (enjoy)
REWARD_LOG_DIR=""${TRAINING_DIR}"/"${ENV_ID}"/reward_logs"

## Extra arguments to be passed into the script
EXTRA_ARGS=""

########################################################################################################################
########################################################################################################################

## Spawn ign_moveit2 subprocess in background, while making sure to forward termination signals
IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2.launch.py"
if [ "$ROBOT_MODEL" = "ur5_rg2" ]; then
    IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_ur5_rg2.launch.py"
fi
if [ "$ROBOT_MODEL" = "kinova_j2s7s300" ]; then
    IGN_MOVEIT2_CMD="ros2 launch drl_grasping ign_moveit2_kinova_j2s7s300.launch.py"
fi
echo "Launching ign_moveit2 in background:"
echo "${IGN_MOVEIT2_CMD}"
echo ""
${IGN_MOVEIT2_CMD} &
## Kill all subprocesses when SIGINT SIGTERM EXIT are received
subprocess_pid_ign_moveit2="${!}"
terminate_subprocesses() {
    echo "INFO: Caught signal, killing all subprocesses..."
    pkill -P "${subprocess_pid_ign_moveit2}"
}
trap 'terminate_subprocesses' SIGINT SIGTERM EXIT ERR

## Arguments
ENJOY_ARGS="--env "${ENV_ID}" --algo "${ALGO}" --seed "${SEED}" --folder "${LOG_DIR}" --reward-log "${REWARD_LOG_DIR}" --env-kwargs "${ENV_ARGS}" "${EXTRA_ARGS}""
## Add trained agent to args in order to continue training
if [ ! -z "${CHECKPOINT}" ]; then
    ENJOY_ARGS=""${ENJOY_ARGS}" --load-checkpoint "${CHECKPOINT}""
fi

## Execute enjoy script
ENJOY_CMD="ros2 run drl_grasping enjoy.py "${ENJOY_ARGS}""
echo "Executing enjoy command:"
echo "${ENJOY_CMD}"
echo ""
${ENJOY_CMD}
