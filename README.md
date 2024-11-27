# Hybrid proximal policy optimization

## Requirements

### Python 3.8  
* Create a virtual environment with python version 3.8  
* Install 2 packages: `pybullet` for simulation and `gymnasium` for reinforcement learning
### `ur_description` package
`git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/humble` then rename this repo to `ur_description`

## Environment
The `environment.py` file implement the environment. 
* Static objects such as table and robotics arm is loaded in the `__init__()` function, meanwhile dynamic objects such as the targets to be pick will be loaded in the `reset()` function.