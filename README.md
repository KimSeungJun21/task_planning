# task_planning
```git clone -b master https://github.com/KimSeungJun21/task_planning.git``` 

```cd task_planning```

```sudo docker build --tag nvidia_ros:latest .```

# Installing nvidia-container-toolkit 
```sudo apt install curl```

```distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list```

```sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit```

```sudo systemctl restart docker```


```sudo docker run --name ros -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ~/task_planning:/workspace --gpus all nvidia_ros /bin/bash```

```cd test_pybullet```

If you want to run examples:

1) Stacking example
  
```python3 stacking.py```

2) Clustering example

```python3 clustering.py```

``` gpus issue =>https://velog.io/@johyonghoon/docker-Error-response-from-daemon-could-not-select-device-driver-with-capabilities-gpu-%ED%95%B4%EA%B2%B0```
