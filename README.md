# task_planning
docker build --tag nvidia_ros:latest .
docker run --name ros -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/starry/Workspace/ksjj_ws:/workspace --gpus all nvidia_ros /bin/bash
rosdep install --from-paths src --ignore-src -y
source devel/setup.bash
catkin_make
roslaunch ur5e_robotiq_moveit demo.launch
