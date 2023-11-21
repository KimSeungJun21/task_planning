# Graph-based Task Planner(GTP)

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
home에 task_planning file이 있는 상태, 만약 다른 폴더에서 실행하는경우, ~/(path)/task_planning으로 경로변경을 해주어야된다.

## ROS setup
### Build packages
```
cd workspace/gtp_ws
catkin_make
source devel/setup.bash
```

## Run with pretrained model
### 1.Run Roscore
```roscore```

### 2.Run pretrained task planner server
#### enter to docker env in new terminal
```
docker exec -it gtp01 bash
```

#### source ros workspace and run pretrained task planner
```
cd ~/workspace/gtp_ws
source devel/setup.bash
cd src/graph_task_planning/src
python plan_inference.py
```

### 3.Run pybullet simulator
#### enter to docker env in new terminal 
```
docker exec -it gtp01 bash
```
#### source ros workspace and run pretrained task planner
```
cd ~/workspace/gtp_ws
source devel/setup.bash
cd src/gtp_pybullet/src
python sim_env.py
```
Now you can see the initial state with 5 boxes randomly located on the white region.
To observe graph-based state and send it to task planner, you need to input [enter] in this terminal when ```request plan...``` message is on it.

## Issue
gpus issue: <https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-%ED%95%B4%EA%B2%B0>
