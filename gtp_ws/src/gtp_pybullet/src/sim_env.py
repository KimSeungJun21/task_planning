#! /usr/bin/env python

import rospy
from math import pi
import os
import logging

import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
import random
import numpy as np
import os
from itertools import permutations

from ros_msg_utils import numpy_to_float64_multiarray

from std_msgs.msg import Float64MultiArray
from gtp_pybullet.msg import Plan_msg
from gtp_pybullet.srv import PlanPrediction, PlanPredictionRequest

logging.basicConfig(level=logging.INFO)

####################################
### Pybullet environment setting ###
####################################
## set pybullet environment parameters ##
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setTimeStep(1/200)
planeID = p.loadURDF("plane.urdf")

## robot setting ##
cubeStartPos=[0,0.24,0.7]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
Robot_ID = p.loadURDF(os.path.join(os.path.dirname(__file__), 'urdf/robots/urdf/ur5e_with_gripper.urdf'),cubeStartPos,cubeStartOrientation,useFixedBase=True,flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
robot_base_ID=p.loadURDF(os.path.join(os.path.dirname(__file__), "./urdf/objects/robot_base.urdf"),[0,0.24,0.35],useFixedBase=True)

## table setting ##
Table_ID = p.loadURDF(os.path.join(os.path.dirname(__file__), "./urdf/objects/table.urdf"),[0,-0.35,0.36],useFixedBase=True)
p.changeVisualShape(Table_ID,-1,rgbaColor=[1,1,1,1])
urdf_path = os.path.join(os.path.dirname(__file__) ,"./urdf/objects/block.urdf")

## box object setting ##
def check_position(x,y,x_previous,y_previous):
    for i in range(len(x_previous)):
        if (x-x_previous[i])<=0.08 and (y-y_previous[i])<=0.07:
            return False
    return True

def position_generation(x_previous,y_previous):
    position_is_unique = False
    while not position_is_unique:
        x = random.uniform(-0.3, 0.3)
        y = random.uniform(-0.6, -0.1)
        position_is_unique = check_position(x,y,x_previous,y_previous)
    return x,y

num_objects = 5
color_list=[[1,0,0,1],[1,1,0,1],[1,1,1,1],[0,1,1,1],[0,0,1,1]]

Box_IDs = []
x_previous=[]
y_previous=[]
for i in range(num_objects):
    x,y=position_generation(x_previous,y_previous)
    x_previous.append(x)
    y_previous.append(y)
    z = 0.8

    object_id = p.loadURDF(urdf_path, [x, y, z])
    Box_IDs.append(object_id)

    p.changeVisualShape(object_id,-1,rgbaColor=color_list[i])
    p.changeDynamics(object_id,-1,lateralFriction=100,restitution=0.2)

## region setting ##
plate_IDs=[]
plate_IDs.append(p.loadURDF(os.path.join(os.path.dirname(__file__),"./urdf/objects/plate.urdf"),[0.45,-0.35,0.72001],useFixedBase=True))
plate_IDs.append(p.loadURDF(os.path.join(os.path.dirname(__file__),"./urdf/objects/plate.urdf"),[-0.45,-0.35,0.72001],useFixedBase=True))
p.changeVisualShape(plate_IDs[1],-1,rgbaColor=[1,0,0,1])

## robot joint setting ##
jointPositions=[-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,-1.5707970583733368, 0.0009377758247187636]

for jointIndex in range(1,7):
    p.resetJointState(Robot_ID,jointIndex,jointPositions[jointIndex-1])
    p.setJointMotorControl2(Robot_ID, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex-1], 0)

arm_joint_ranges=[p.getJointInfo(Robot_ID,i)[9] - p.getJointInfo(Robot_ID,i)[8] for i in range(p.getNumJoints(Robot_ID)) if p.getJointInfo(Robot_ID,i)[2]!=p.JOINT_FIXED][1:7]
arm_rest_poses=[-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636]

###############################
### graph state observation ###
###############################
def switch_node_ids(Box_IDs, Robot_ID, Table_ID, plate_IDs):
    object_ids = []

    object_ids.extend(Box_IDs)
    object_ids.append(Robot_ID)
    object_ids.append(Table_ID)
    object_ids.extend(plate_IDs)
    # object_ids = [4, 5, 6, 7, 8, 1, 3, 9, 10]

    node_id = {number : index  for index, number in enumerate(object_ids)}
   
    return node_id

def node_feature(node_id):
    result_list = []

    for index in node_id.values():
        tensor = np.zeros(len(node_id) + 3)
        tensor[index] = 1

        if 0 <= index < 5:
            tensor[-3] = 1
        elif index == 5:
            tensor[-2] = 1
        else:
            tensor[-1] = 1

        result_list.append(tensor)
    node_feature = np.vstack(result_list)

    return node_feature

def edge_feature(node_id):
    # node_id = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 1: 5, 3: 6, 9: 7, 10: 8}
    pose_ids = [p.getBasePositionAndOrientation(id) for id in node_id.keys()] # 9 ids
    combination_pair = list(permutations(pose_ids, 2)) #90 length
    idx_com_pairs = list(permutations(node_id.values(), 2)) 

    # Edge index
    # 2d tensor
    edge_attr = np.zeros((len(idx_com_pairs), 4), int)

    for num in range(len(combination_pair)): 
        obj1_pose = combination_pair[num][0]
        obj2_pose = combination_pair[num][1]

        obj1_id = [index for index, pose_id in enumerate(pose_ids) if pose_id == obj1_pose][0]
        obj2_id = [index for index, pose_id in enumerate(pose_ids) if pose_id == obj2_pose][0]

        obj1_x = obj1_pose[0][0]
        obj2_x = obj2_pose[0][0]

        obj1_y = obj1_pose[0][1]
        obj2_y = obj2_pose[0][1]

        obj1_z = obj1_pose[0][2]
        obj2_z = obj2_pose[0][2]
        # print(combination_pair, obj1_z, obj2_z)

        # Digraph 
        # on_right(box table), on_left, grasp_right(hand box), grasp_left
        
        for inx, pair in enumerate(idx_com_pairs): 
            if 0 <= obj1_id <= 4:
                # Box - Table 
                # Region 6 -> White
                if -0.3 < obj1_x < 0.3 and -0.7 <= obj1_y <= 0 and 0.69 <= obj1_z <= 0.77:
                    if pair == (obj1_id, 6):
                        edge_attr[inx][0] = 1
                    if pair == (6, obj1_id):
                        edge_attr[inx][1] = 1
                    # print('on_right', obj1_id, '6')
                # Region 7 -> Blue
                if 0.3 <= obj1_x <= 0.6 and -0.7 <= obj1_y <= 0 and 0.69 <= obj1_z <= 0.77:
                    # print('on_right', obj1_id, '7')
                    if pair == (obj1_id, 7):
                        edge_attr[inx][0] = 1
                    if pair == (7, obj1_id):
                        edge_attr[inx][1] = 1
                # Region 8 -> Red
                if -0.6 <= obj1_x <= -0.3 and -0.7 <= obj1_y <= 0 and 0.69 <= obj1_z <= 0.77:
                    # print('on_right', obj1_id, '8')
                    if pair == (obj1_id, 8):
                        edge_attr[inx][0] = 1
                    if pair == (8, obj1_id):
                        edge_attr[inx][1] = 1
                
                # Robot - box 
                gripper_pose = p.getLinkState(Robot_ID, 10)[0]
                gripper_pose_x = gripper_pose[0]
                gripper_pose_y = gripper_pose[1]
                gripper_pose_z = gripper_pose[2]

                if -0.3 <= obj1_x - gripper_pose_x <= 0.3 and -0.3 <= obj1_y - gripper_pose_y <= 0.3 and -0.1 <= obj1_z - gripper_pose_z <= 0.1:
                    # print('grasp_right', obj1_id, 5)
                    if pair == (5, obj1_id):
                        edge_attr[inx][2] = 1
                    if pair == (obj1_id, 5):
                        edge_attr[inx][3] = 1

                # Box - Box
                if 0 <= obj2_id <= 4 and obj1_id != obj2_id:
                    if -0.1 <= obj1_x - obj2_x <= 0.1 and -0.1 <= obj1_y - obj2_y <= 0.1 and 0 <= obj1_z - obj2_z <= 0.1: 
                        # print('on_right (stack)', obj1_id, obj2_id)
                        if pair == (obj1_id, obj2_id):
                            edge_attr[inx][0] = 1
                        if pair == (obj2_id, obj1_id):
                            edge_attr[inx][1] = 1

    edge_index = [[idx_com_pairs[index][0],idx_com_pairs[index][1]] for index, attr in enumerate(edge_attr) if not np.all(attr == 0)]
    edge_index = np.array(edge_index)
    edge_index = edge_index.T

    edge_attr = [attr for attr in edge_attr if not np.all(attr == 0)]
    edge_attr = np.array(edge_attr)

    # print(edge_index)
    # print(edge_attr)

    return edge_index, edge_attr

################################
### pybullet action function ###
################################
def step_simulation():
    p.stepSimulation()
    time.sleep(1/240)

def __setup_mimic_joints__(gripper_main_control, mimic_joint_name):
    mimic_parent_id = p.getJointInfo(Robot_ID,10)[0] 
    mimic_child_multiplier = {p.getJointInfo(Robot_ID,joint)[0]: mimic_joint_name[p.getJointInfo(Robot_ID,joint)[1].decode("utf-8")] for joint in range(p.getNumJoints(Robot_ID)) if p.getJointInfo(Robot_ID,joint)[1].decode("utf-8") in mimic_joint_name}

    for joint_id, multiplier in mimic_child_multiplier.items():
        c = p.createConstraint(Robot_ID, mimic_parent_id,
                               Robot_ID, joint_id,
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


class SimExecutor:
    def __init__(self):
        self.delay_time = 10
        self.joint_states = None
        self.pose_initialize()

        self.obj_id_dict = None
        self.gripper_state = None

    def pose_initialize(self):
        # move to initial pose
        self.joint_states=[0,-0.3,1,0,1.57,1.57,0.085]

        for _ in range(self.delay_time):
            self.step('end')
    
    def step(self,control_method='joint'):
        self.move_ee(control_method)
        self.move_gripper()
        __post_load__()
        for _ in range(120):
            step_simulation()

    def move_ee(self, control_method):
        action = self.joint_states[:-1]
        #print('action:',action)
        #print('control_method:',control_method)

        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z+0.15)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(Robot_ID, 9, pos, orn, arm_joint_ranges, arm_rest_poses,maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == 6
            joint_poses = action
        # arm
        for i, joint_id in enumerate([1,2,3,4,5,6]):
            p.setJointMotorControl2(bodyIndex=Robot_ID, jointIndex=joint_id, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],maxVelocity=p.getJointInfo(Robot_ID,joint_id)[11]/15)

    def move_gripper(self):
        open_length = self.joint_states[-1]
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(Robot_ID, p.getJointInfo(Robot_ID,10)[0], p.POSITION_CONTROL, targetPosition=open_angle,targetVelocity=p.getJointInfo(Robot_ID,10)[11],force=p.getJointInfo(Robot_ID,10)[10]/5)


    def pre_grasp_pose(self, obj_id):
        self.joint_states[0]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][0]
        self.joint_states[1]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][1]
        self.joint_states[2]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][2]+0.15
        rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[1])
        self.joint_states[5]=rz

        for _ in range(self.delay_time):
            self.step('end')
    

    def move_box(self, obj_id):
        self.joint_states[0]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][0]
        self.joint_states[1]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][1]
        self.joint_states[2]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][2]
        rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[1])
        self.joint_states[5]=rz
        
        for _ in range(self.delay_time):
            self.step('end')
    

    def gripper_state_control(self, gripper_action, obj_id):
        if gripper_action=='close' and self.gripper_state==None:
            cal=(p.getLinkState(Robot_ID,9)[0][2]-p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][2])
            self.gripper_state=p.createConstraint(Robot_ID, p.getJointInfo(Robot_ID,9)[0], self.obj_id_dict[obj_id], -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [-cal+0.0001, 0, 0])

        elif gripper_action=='open' and self.gripper_state!=None:
            p.removeConstraint(self.gripper_state)
            self.gripper_state=None

    def gripper_close(self, gripper_length, obj_id):
        self.gripper_state_control('close', obj_id)
        time.sleep(0.1)
        self.joint_states[6]=gripper_length

        for _ in range(self.delay_time):
            self.step('end')
    

    def gripper_open(self, obj_id):        
        self.gripper_state_control('open', obj_id)
        time.sleep(0.1)
        self.joint_states[6]=0.085

        for _ in range(self.delay_time):
            self.step('end')
    

    def move_xyz(self, direction, length):
        if direction == 'x':
            self.joint_states[0]=length
        elif direction == 'y':
            self.joint_states[1]=length
        elif direction == 'z':
            self.joint_states[2]=length

        for _ in range(self.delay_time):
            self.step('end')
    

    def stack(self, obj_id):
        self.joint_states[0]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][0]
        self.joint_states[1]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][1]
        self.joint_states[2]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][2]+0.08
        rx,ry,rz=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[1])
        self.joint_states[5]=rz
        
        for _ in range(self.delay_time):
            self.step('end')
    

    def move_plate(self, obj_id):
        if obj_id > 6:
            self.joint_states[0]=np.random.normal(p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][0],0.03)
            self.joint_states[1]=np.random.normal(p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][1],0.05)
            self.joint_states[2]=p.getBasePositionAndOrientation(self.obj_id_dict[obj_id])[0][2]+0.1
        elif obj_id==6:
            self.joint_states[0]=np.random.normal(0,0.03)
            self.joint_states[1]=np.random.normal(-0.35,0.05)
            self.joint_states[2]=0.82001

        for _ in range(self.delay_time):
            self.step('end')
    

if __name__ == "__main__":
    rospy.init_node('pybullet_env')

    # start ros client
    rospy.wait_for_service("/pred_plan")
    plan_client = rospy.ServiceProxy("/pred_plan", PlanPrediction)
    
    request_srv = PlanPredictionRequest()
    # object id dict
    obj_id_dict = switch_node_ids(Box_IDs, Robot_ID, Table_ID, plate_IDs)
    executor = SimExecutor()
    executor.obj_id_dict = {v:k for k,v in obj_id_dict.items()}
    act_name_list = ['pick', 'place']
    obj_name_list = ['box1', 'box2', 'box3', 'box4', 'box5', 'robot_hand', 'region_mid', 'region_blue', 'region_red']

    goal_reached = False
    plan_sequence = []
    while not goal_reached:
        # input("request plan...")
        print("request plan...")

        nf = node_feature(obj_id_dict)
        ei, ea = edge_feature(obj_id_dict)
        # print(nf, ei, ea)

        nf = numpy_to_float64_multiarray(nf)
        ei = numpy_to_float64_multiarray(ei)
        ea = numpy_to_float64_multiarray(ea)
        # print(nf, ei, ea)

        request_srv.node_feature = nf
        request_srv.edge_index = ei
        request_srv.edge_attr = ea

        result = plan_client(request_srv)
        
        pred_action = result.pred_plan.action
        pred_object = result.pred_plan.object
        print('pred_plan:\n')
        print('  action: ', act_name_list[pred_action])
        print('  object: ', obj_name_list[pred_object])
        
        if pred_action == -1 and pred_object == -1:
            goal_reached = True
            break
        plan_sequence.append((act_name_list[pred_action], obj_name_list[pred_object]))

        ## action execution ##
        if pred_action == 0: #pick
            executor.pre_grasp_pose(pred_object)
            executor.move_box(pred_object)
            executor.gripper_close(0.06,pred_object)
            executor.move_xyz('z', 1)

        elif pred_action == 1: #place
            # check if pred_object is box
            if pred_object >= 5: # robot or region type
                executor.move_plate(pred_object)
            else: # box type
                executor.stack(pred_object)
            executor.gripper_open(pred_object)
            executor.move_xyz('z', 1)
            executor.pose_initialize()
        else:
            print('error')
        
        
    print('goal_reached!')
    print(plan_sequence)
    input()
