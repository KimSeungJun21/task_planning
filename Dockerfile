FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN echo 'export PATH=/usr/local/cuda-11.6.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.6.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:/$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
RUN apt-get -y update
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata 
# 타임존 미리 설정
RUN apt-get -y update
#RUN apt-get install -y gedit 
#RUN apt-get install -y gedit
RUN apt-get install -y vim
RUN apt-get install -y sudo
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y tesseract-ocr tesseract-ocr-kor
RUN apt-get install -y x11-apps

# Install conda
#RUN conda init bash
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN sudo apt install -y python3
RUN sudo apt install -y python3-pip
# Install pybullet
RUN pip install numpy
RUN pip install pybullet
# Install ROS
#RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#RUN sudo apt-get install -y curl 
#RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
#RUN sudo apt update
#RUN sudo apt install -y ros-noetic-desktop-full
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
#RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
#RUN /bin/bash -c "source ~/.bashrc"
#RUN sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
#RUN sudo apt install -y python3-rosdep
#RUN sudo rosdep init
#RUN rosdep update 
#RUN sudo apt-get update
#RUN sudo apt-get install -y software-properties-common
#RUN sudo apt-get install -y ros-noetic-moveit-setup-assistant
#RUN sudo apt-get install -y ros-noetic-moveit-core ros-noetic-moveit-ros-planning ros-noetic-moveit-ros-planning-interface ros-noetic-moveit-ros-robot-interaction
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-moveit-visual-tools
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-speed-scaling-interface
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-speed-scaling-state-controller
#RUN sudo apt-get install -y ros-noetic-ur-msgs
#RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#RUN sudo apt install -y curl 
#RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
#RUN sudo apt update
#RUN sudo apt install -y ros-noetic-desktop-full
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
#RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
#RUN sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
#RUN sudo apt install -y python3-rosdep
#RUN sudo rosdep init
#RUN rosdep update
RUN sudo apt-get update
RUN sudo apt-get install -y software-properties-common


#RUN mkdir workspace
WORKDIR /workspace
RUN mkdir src
# Set up ROS environment and install rosdep

#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && rosdep update && rosdep install --from-paths /workspace/src --ignore-src -y"
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-rviz-visual-tools
#RUN sudo apt-get install -y ros-noetic-moveit-fake-controller-manager ros-noetic-moveit-planners ros-noetic-moveit-simple-controller-manager \
#    && sudo apt-get install -y ros-noetic-position-controllers \
#    && sudo apt-get install -y ros-noetic-joint-trajectory-controller
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-scaled-joint-trajectory-controller
#RUN sudo apt-get update && sudo apt-get install -y ros-noetic-industrial-robot-status-interface
#RUN rosdep install -y --from-paths /workspace/src/
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3"

#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && rosdep update && rosdep install --from-paths /workspace/src --ignore-src -y"
#RUN rosdep install -y --from-paths /workspace/src/
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3"

