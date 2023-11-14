# task_planning
```git clone -b master https://github.com/KimSeungJun21/task_planning.git``` 

```cd task_planning```

```sudo docker build --tag nvidia_ros:latest .```

```docker run --name ros -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v ~/task_planning:/workspace --gpus all nvidia_ros /bin/bash```

```cd test_pybullet```

If you want to run examples:

1) Stacking example
  
```python3 stacking.py```

2) Clustering example

```python3 clustering.py```
