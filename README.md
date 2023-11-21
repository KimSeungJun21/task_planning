# task_planning
```
git clone -b master https://github.com/KimSeungJun21/task_planning.git

cd task_planning

xhost +
```

# Docker build
```sudo docker build --tag gtp_env .```

```sudo docker run --privileged --name gtp01 -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ~/task_planning:/workspace --gpus all gtp_env:latest /bin/bash```

# ROS package build
```cd workspace/gtp_ws```
```catkin_make```
```source devel/setup.bash```
# Run
## 1.Run Roscore

```roscore```

## 2.Run task planner server

```cd src/graph_task_planning/src```

```python plan_inference.py```

## 3.Run pybullet simulator

```cd src/gtp_pybullet/src```

```python sim_env.py```

# Installing nvidia-container-toolkit
```sudo apt install curl```

```distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list```

```sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit```

```sudo systemctl restart docker```

```home에 task_planning file이 있는 상태, 만약 다른 폴더에서 실행하는경우, ~/path/task_planning으로 경로변경을 해주어야된다. ```

```sudo docker run --name ros -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ~/task_planning:/workspace --gpus all nvidia_ros /bin/bash```

```cd test_pybullet```

If you want to run examples:

1) Stacking example
  
```python3 stacking.py```

2) Clustering example

```python3 clustering.py```

``` gpus issue =>https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-%ED%95%B4%EA%B2%B0```
