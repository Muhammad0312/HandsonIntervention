import math
from math import sin, cos
import numpy as np
from utils_lib.functions_robotics import *
import tf

'''
    Class representing a robotic manipulator.
'''
class MobileManipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self,name,l1 = 0.108, l2 = 0.142, l3 = 0.1588, l4  = 0.0565, l5  = 0.0722):

        self.name = name
        self.l1 = l1  
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.T = np.zeros((6,))
        self.J = np.zeros((6,6))

        self.q1 = 0
        self.q2 = 0
        self.q3 = 0
        self.q4 = 0

        self.x_base = 0
        self.y_base = 0
        self.theta_base = 0

        self.alpha = 0
        self.d = 0

    def getJointPos(self, joint):
        return self.q[joint]
    
    def update(self,t1,t2,t3,t4,x_base,y_base,theta_base,alpha,d):

        # update kinematics
        self.T = kinematics_total(t1, t2, t3,t4, self.l1, self.l2, self.l3, self.l4, self.l5, \
                                    x_base, y_base, theta_base, alpha, d)
        
        # update jacobians
        self.J = jacobian_total(t1, t2, t3, self.l1, self.l2, self.l3, self.l4, self.l5, \
                                    x_base, y_base, theta_base, alpha, d)
        
        [self.q1, self.q2, self.q3, self.q4] = [t1, t2, t3, t4]

        [self.x_base, self.y_base, self.theta_base] = [x_base, y_base, theta_base]
        
        [self.alpha, self.d] = [alpha, d]
        
    def get_T(self):
        return np.array(self.T)

    def get_J(self):
        return self.J
    
    
'''
    Base class representing an abstract Task.
'''
class Task:
    def __init__(self, name, desired=None,link=-1):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.error = None
        self.K = None
        self.link = link        
        self.Active = False

    def isActive(self):
        return self.Active

    def setDesired(self, value):
        self.sigma_d = value

    def getDesired(self):
        return self.sigma_d

    def getJacobian(self):
        return self.J

    def getError(self):
        return self.err
        
    def setK(self,K):
        self.K = K

    def getK(self):
        return self.K
    
'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired,link=-1):
        super().__init__(name, desired,link)
        self.J = np.zeros((3, 6)) # 2D pos- 3 joints 
        self.err = np.zeros(3) # Initialize with proper dimensions (2 elements)
        self.Active = True
        
    def update(self, robot):
        # Compute the current position of the end-effector
        
        current_pos = robot.get_T()[:3].reshape(3,1)

        self.err = self.getDesired() - current_pos

        self.J = robot.get_J()[:3,:]

        return self.J, self.err

class Orientation2D(Task):
    def __init__(self, name, desired, link=-1):
        super().__init__(name, desired,link)
        self.J = np.zeros((3, 6)) # Initialize with proper dimensions
        self.err = np.zeros((3,1)) # Initialize with proper dimensions
        self.Active = True

    def update(self, robot):
        # get desired orientation in rpy
        desried_orient = self.getDesired()  # give a 3x1 rpy matrix
        
        # change desired orient to quartinion
        quart_desired_orient = rpy_to_quaternion(desried_orient)
        w_d = quart_desired_orient[0]
        e_d = np.array([quart_desired_orient[1],quart_desired_orient[2],quart_desired_orient[3]])

        # # Getting the current orient
        curr_orient = robot.get_T()[3:].reshape(3,1)
        quart_orient = rpy_to_quaternion(curr_orient)
        w = quart_orient[0]
        e = np.array([quart_orient[1],quart_orient[2],quart_orient[3]])

        # # Compute the error for the orientation task
        self.err = w * e_d - w_d * e - np.cross(e,e_d)
        self.err = self.err.reshape(3,1) 

        # # Update the task Jacobian
        self.J = robot.get_J()[3:,:]

        return self.J, self.err
    
class Configuration(Task):
    def __init__(self, name, desired,link=-1):
        super().__init__(name, desired,link)
        self.J = np.zeros((6, 6)) # 2D pos- 3 joints 
        self.err = np.zeros((6,1)) # Initialize with proper dimensions (2 elements)
        self.Active = True
        
    def update(self, robot):
        # Compute the current position of the end-effector
        
        current_pos = robot.get_T().reshape(6,1)

        self.err = self.getDesired() - current_pos

        self.J = robot.get_J()

        return self.J, self.err
    
class BaseOrientation(Task):
    def __init__(self, name, desired,link=-1):
        super().__init__(name, desired,link)
        self.J = np.zeros((1, 6)) # 2D pos- 3 joints 
        self.err = np.zeros((1,1)) # Initialize with proper dimensions (2 elements)
        self.Active = True
        
    def update(self, robot):
        # Compute the current position of the end-effector
        
        current_pos = robot.theta_base 

        self.err = self.getDesired() - current_pos

        self.J = np.array([0,0,0,0,1,0]).reshape(1,6)

        return self.J, self.err
    
''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired,link=-1):
        super().__init__(name, desired,link)
        self.J = np.zeros((1, 6))
        # print('J in innit: ', self.J)
        self.err = 0
        self.Active = True
        
    def update(self, robot):

        self.J[0,self.link] = 1.0

        joints = [robot.q1, robot.q2, robot.q3, robot.q4]
        joint_pos = joints[self.link]

        self.err = self.sigma_d - joint_pos

        return self.J, self.err
        

''' 
    Subclass of Task, representing joint  task.
'''
class JointLimit(Task):
    def __init__(self, name, q_min,q_max, q_tol,joint_index):
        super().__init__(name)
        self.J = np.zeros((1, 6))
        self.err = np.array([0]) #0
        self.q_min = q_min
        self.q_max = q_max
        self.q_tol = q_tol

        self.link = joint_index

    def update(self, robot):

        joints = [robot.q1, robot.q2, robot.q3, robot.q4]
        joint_pos = joints[self.link]

        self.J[0,self.link] = 1.0

        if joint_pos < self.q_min + self.q_tol:
            # self.err = joint_pos - [self.q_max]
            self.err = np.array([1.0])
            self.Active = True
        elif joint_pos > self.q_max - self.q_tol:
            # self.err = [self.q_min] - joint_pos
            self.err = np.array([-1.0])
            self.Active = True
        else:
            self.Active = False

        return self.J, self.err

