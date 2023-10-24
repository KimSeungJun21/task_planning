import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
import random
import numpy as np

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setTimeStep(1/200)
planID = p.loadURDF("plane.urdf")
cubeStartPos=[0,0.24,0.7]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotID = p.loadURDF("./urdf/robots/urdf/ur5e_with_gripper.urdf",cubeStartPos,cubeStartOrientation,useFixedBase=True,flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
robot_base_ID=p.loadURDF("./urdf/objects/robot_base.urdf",[0,0.24,0.35],useFixedBase=True)
table_id = p.loadURDF("./urdf/objects/table.urdf",[0,-0.35,0.36],useFixedBase=True)
p.changeVisualShape(table_id,-1,rgbaColor=[1,1,1,1])
urdf_path = "./urdf/objects/block.urdf"
gripper_state=None

# 생성할 물체의 개수
num_objects = 5
color_list=[[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,0,1,1],[0,0,1,1]]
# 물체 생성 루프
object_ids = []
for i in range(num_objects):
    x = random.uniform(-0.3, 0.3)
    y = random.uniform(-0.4, -0.1)
    z = 0.8

    object_id = p.loadURDF(urdf_path, [x, y, z])
    object_ids.append(object_id)
for i in range(num_objects):
    p.changeVisualShape(object_ids[i],-1,rgbaColor=color_list[i])
    p.changeDynamics(object_ids[i],-1,lateralFriction=100,restitution=0.2)

plate_ID=[]
plate_ID.append(p.loadURDF("./urdf/objects/plate.urdf",[0.45,-0.35,0.72001],useFixedBase=True))
plate_ID.append(p.loadURDF("./urdf/objects/plate.urdf",[-0.45,-0.35,0.72001],useFixedBase=True))
p.changeVisualShape(plate_ID[1],-1,rgbaColor=[1,0,0,1])

jointPositions=[-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,-1.5707970583733368, 0.0009377758247187636]

for jointIndex in range(1,7):
    p.resetJointState(robotID,jointIndex,jointPositions[jointIndex-1])
    p.setJointMotorControl2(robotID, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex-1], 0)

gripper_range=[0,0.04]

#c = p.createConstraint(robotID,9,robotID,10,jointType=p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
#p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
arm_joint_ranges=[p.getJointInfo(robotID,i)[9] - p.getJointInfo(robotID,i)[8] for i in range(p.getNumJoints(robotID)) if p.getJointInfo(robotID,i)[2]!=p.JOINT_FIXED][1:7]
arm_rest_poses=[-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636]

def step_simulation():
    p.stepSimulation()
    time.sleep(1/240)

def __setup_mimic_joints__(gripper_main_control, mimic_joint_name):
    mimic_parent_id = p.getJointInfo(robotID,10)[0] 
    mimic_child_multiplier = {p.getJointInfo(robotID,joint)[0]: mimic_joint_name[p.getJointInfo(robotID,joint)[1].decode("utf-8")] for joint in range(p.getNumJoints(robotID)) if p.getJointInfo(robotID,joint)[1].decode("utf-8") in mimic_joint_name}

    for joint_id, multiplier in mimic_child_multiplier.items():
        c = p.createConstraint(robotID, mimic_parent_id,
                               robotID, joint_id,
                                jointType=p.JOINT_GEAR,
                                jointAxis=[0, 1, 0],
                                parentFramePosition=[0, 0, 0],
                                childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-multiplier, maxForce=1000, erp=1) 

def __post_load__():
    # To control the gripper
    gripper_main_control_joint_name = 'robotiq_85_left_knuckle_joint'
    mimic_joint_name = {'robotiq_85_right_knuckle_joint':1 ,
                            'robotiq_85_left_inner_knuckle_joint':1,
                            'robotiq_85_right_inner_knuckle_joint':1,
                            'robotiq_85_left_finger_tip_joint':-1,
                            'robotiq_85_right_finger_tip_joint':-1}
    #mimic_multiplier=[1,1,1,-1,-1]
    __setup_mimic_joints__(gripper_main_control_joint_name, mimic_joint_name)








def move_gripper(open_length):
    # open_length = np.clip(open_length, *self.gripper_range)
    open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
    # Control the mimic gripper joint(s)
    p.setJointMotorControl2(robotID, p.getJointInfo(robotID,10)[0], p.POSITION_CONTROL, targetPosition=open_angle,targetVelocity=p.getJointInfo(robotID,10)[11],force=p.getJointInfo(robotID,10)[10]/5)



def step(action,control_method='joint'):
    move_ee(action[:-1],control_method)
    move_gripper(action[-1])
    __post_load__()
    for _ in range(120):
        step_simulation()


def move_ee(action, control_method):
    #print('action:',action)
    #print('control_method:',control_method)

    assert control_method in ('joint', 'end')
    if control_method == 'end':
        x, y, z, roll, pitch, yaw = action
        pos = (x, y, z+0.15)
        orn = p.getQuaternionFromEuler((roll, pitch, yaw))
        joint_poses = p.calculateInverseKinematics(robotID, 7, pos, orn, p.getJointInfo(robotID,6),p.getJointInfo(robotID,7), arm_joint_ranges, arm_rest_poses,maxNumIterations=20)
    elif control_method == 'joint':
        assert len(action) == 6
        joint_poses = action
    # arm
    for i, joint_id in enumerate([1,2,3,4,5,6]):
        p.setJointMotorControl2(bodyIndex=robotID, jointIndex=joint_id, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],maxVelocity=p.getJointInfo(robotID,joint_id)[11]/15)

def move_box(action,i):
    action[0]=p.getBasePositionAndOrientation(object_ids[i])[0][0]
    action[1]=p.getBasePositionAndOrientation(object_ids[i])[0][1]
    action[2]=p.getBasePositionAndOrientation(object_ids[i])[0][2]
    rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(object_ids[0])[1])
    action[5]=rz
    return action

def move_plate(action,j):
    if j <3:
        action[0]=np.random.normal(p.getBasePositionAndOrientation(plate_ID[1])[0][0],0.02)
        action[1]=np.random.normal(p.getBasePositionAndOrientation(plate_ID[1])[0][1],0.02)
        action[2]=p.getBasePositionAndOrientation(plate_ID[1])[0][2]+0.1
    else:
        action[0]=np.random.normal(p.getBasePositionAndOrientation(plate_ID[0])[0][0],0.02)
        action[1]=np.random.normal(p.getBasePositionAndOrientation(plate_ID[0])[0][1],0.02)
        action[2]=p.getBasePositionAndOrientation(plate_ID[0])[0][2]+0.1
        
    return action

def move_xyz(action,str,i):
    if str == 'x':
        action[0]=i
    elif str == 'y':
        action[1]=i
    elif str == 'z':
        action[2]=i
    return action

def pre_grasp_pose(action,i):
    action[0]=p.getBasePositionAndOrientation(object_ids[i])[0][0]
    action[1]=p.getBasePositionAndOrientation(object_ids[i])[0][1]
    action[2]=p.getBasePositionAndOrientation(object_ids[i])[0][2]+0.1
    rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(object_ids[i])[1])
    action[5]=rz
    return action

def stack(action,i):
    action[0]=p.getBasePositionAndOrientation(object_ids[i])[0][0]
    action[1]=p.getBasePositionAndOrientation(object_ids[i])[0][1]
    action[2]=p.getBasePositionAndOrientation(object_ids[i])[0][2]+0.02
    rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(object_ids[i])[1])
    action[5]=rz
    return action    

def gripper_state_control(j,i):
    global gripper_state
    if j==1 and gripper_state==None:
        gripper_state=p.createConstraint(robotID, p.getJointInfo(robotID,7)[0], object_ids[i], -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [-0.1438, 0, 0])
    elif j==0 and gripper_state!=None:
        p.removeConstraint(gripper_state)
        gripper_state=None

def gripper_close(action,range,i):
    gripper_state_control(1,i)
    time.sleep(0.1)
    action[-1]=range
    return action 

def gripper_open(action,i):        
    gripper_state_control(0,i)
    time.sleep(0.1)
    action[-1]=0.085  
    return action

def initial_pose():
    action=[0,-0.3,1,0,1.57,1.57,0.085]
    return action

def clustering(action,i,j):
    if i == 10 + 90*j :
        action=pre_grasp_pose(action,j)
    elif i==20 + 90*j:
        action=move_box(action,j)
    elif i==30 + 90*j:
        action=gripper_close(action,0.06,j)
    elif i==40 + 90*j:
        action=move_xyz(action,'z',1)
    elif i==50 + 90*j:    
        action=move_plate(action,j)
    elif i==60+ 90*j:    
        stack(action,j)
    elif i==70 + 90*j:
        action=gripper_open(action,j)
    elif i ==80 + 90*j:
        action=move_xyz(action,'z',1)
    elif i==90 + 90*j:
        action=initial_pose()
        j+=1
    
    return action,j
action=[0,-0.3,1,0,1.57,1.57,0.085]

j=0
for i in range(10000):
    print('i',i)
    step(action,'end')
    clustering(action,i,j)
p.disconnect(physicsClient)