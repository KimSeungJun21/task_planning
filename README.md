# Graph-based Task Planner (GTP)

![Screenshot from 2023-11-21 11-04-13](https://github.com/KimSeungJun21/task_planning/assets/120440095/a8bcfa5b-53f2-4099-adad-073eb090162b)


## Installation

### Installing nvidia-container-toolkit
```
sudo apt install curl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Environment setup
#### Clone the repository
```
git clone -b master https://github.com/KimSeungJun21/task_planning.git
cd task_planning
```

#### Setup docker environment
```
sudo docker build --tag gtp_env .
sudo docker run --privileged --name gtp01 -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ~/task_planning:/workspace --gpus all gtp_env:latest /bin/bash
```

If you already have task_planning file in home, you should replace the repository to '~/(path)/task_planning'.

## ROS setup
### Build packages
```
cd workspace/gtp_ws
catkin_make
source devel/setup.bash
```

## Run with pretrained model
### 1. Run Roscore
```
roscore
```

### 2. Run pretrained task planner server
#### Enter to docker env in new terminal
```
docker exec -it gtp01 bash
```

#### Source ros workspace and run pretrained task planner
```
cd ~/workspace/gtp_ws
source devel/setup.bash
cd src/graph_task_planning/src
python plan_inference.py
```
You have to give goal conditions to planner. There are 3 conditions: [Task_type, box_order, region_list]

  **task_type**: the task type to plan ['stacking', 'clustering']
  
  **box_order**: the order of boxes to stack(only for stacking): [permutation of 1~5]
  
  **region_list**: the list of regions(red/blue) to cluster 5 boxes(only for clustering): [combination of [r, b] 5 times in total]

After setting goal conditions, the planner server will be started and wait the state input from simulation node.

### 3. Run pybullet simulator
#### Enter to docker env in new terminal 
```
docker exec -it gtp01 bash
```
#### Source ros workspace and run pretrained task planner
```
cd ~/workspace/gtp_ws
source devel/setup.bash
cd src/gtp_pybullet/src
python sim_env.py
```
Now you can see the initial state with 5 boxes randomly located on the white region.
The simulation node will observe graph-based state and send it to task planner sequentially untill the current state is reach the goal state.
## Issue
gpus issue: <https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-%ED%95%B4%EA%B2%B0>
