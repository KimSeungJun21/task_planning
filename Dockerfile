FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get -y update
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata # 타임존 미리 설정
RUN apt-get -y update

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
# Install pybullet
RUN pip install numpy
RUN pip install pybullet
# Install ROS # apt-get으로 수정
RUN apt-get update
RUN apt-get install lsb -y # 추가
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get install -y curl 
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get clean && apt-get update
RUN apt-get install -y ros-noetic-desktop
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN rosdep init
RUN rosdep update
RUN apt-get update
RUN apt-get install -y software-properties-common

# Install pytorch-geometric
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch_geometric

# Install additional packages
RUN apt-get install python-is-python3
RUN pip install natsort
RUN pip install pandas
RUN pip install rospkg
RUN pip install matplotlib
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
#RUN apt-get install -y tesseract-ocr tesseract-ocr-kor
#RUN apt-get install -y x11-apps


#RUN mkdir workspace
#WORKDIR /workspace
#RUN mkdir src
# Set up ROS environment and install rosdep
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && rosdep update && rosdep install --from-paths /workspace/src --ignore-src -y"
RUN rosdep install -y --from-paths /workspace/src/
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python"
#RUN rosdep install -y --from-paths /workspace/src/
#RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3"

