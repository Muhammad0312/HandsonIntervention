#!/usr/bin/env python3
import rospy
import time
import tf
import numpy as np
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg  import JointState
from math import cos, sin, tan
from numpy.linalg import pinv
from autonomous_task_NAK.srv import intervention_getpoint
from utils_lib.task_classes import *
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerRequest, EmptyResponse, TriggerResponse


class JointController:
    def __init__(self):

        self.sigma_d = [0.0,0.0,0.0,0,0,0]
        self.damping = 0.1

        self.desired_received = False
        self.goal_reached = False

        # self.weights = np.diag([0.35, 1.0 , 1.0, 1.0, 5.0, 1.0])
        # self.weights = np.diag([0.2, 1.0 , 1.0, 1.0, 2.0, 2.0]) # working
        # self.weights = np.diag([1.0, 1.0 , 1.0, 1.0, 0.2, 1.0])
        self.weights = np.diag([1.0, 1.0 , 1.0, 1.0, 1.0, 1.0])

        # task related
        self.robot =  MobileManipulator("NAK-Bot")
        # self.tasks = [Position2D("End-effector position", np.array([1.0,1.0,-0.2]).reshape(3,1))] 
        # self.tasks = [Position2D("End-effector position", np.array([0.2,0.2,-0.2]).reshape(3,1)),
        #               Orientation2D("End-effector orientation", np.array([0,0,1.57]).reshape(3,1))] 
        # self.tasks = [Orientation2D("End-effector orientation", np.array([0,0,1.57]).reshape(3,1))]
        # ,\
        #               Position2D("End-effector position", np.array([0.2,0.2,-0.2]).reshape(3,1))] 

        # self.tasks = [Configuration("End-effector configuration", np.array(self.sigma_d).reshape(6,1))]
        # self.tasks = [Position2D("End-effector position", np.array([0.2,0.2,-0.2]).reshape(3,1)),
        #               BaseOrientation("Base Orientation", np.array([-1.57]).reshape(1,1)),
        #               ] JointPosition

        # self.tasks = [JointPosition("Joint position", np.array([1.57]).reshape(1,1),0),
        #               Position2D("End-effector position", np.array([0.2,0.2,-0.2]).reshape(3,1)),
        #               JointPosition("Joint position", np.array([1.57]).reshape(1,1),3)
        #               ]

        # self.tasks = [JointLimit("Joint limit", -1.57, 1.57, 0.5, 0),
        self.tasks = [JointLimit("Joint limit", -1.57, 1.57, 0.2, 0),
                      JointLimit("Joint limit", -1.57, 0, 0.2, 1),
                      JointLimit("Joint limit", -1.57, 0, 0.2, 2),
                      JointLimit("Joint limit", -1.57, 1.57, 0.2, 3),
                      Position2D("End-effector position", np.array([0.2,0.2,0.5]).reshape(3,1))]
        
        # self.tasks = [Position2D("End-effector position", np.array([0.2,0.2,0.1]).reshape(3,1))]
                      
                      

        # Set value of K for all tasks
        for t in self.tasks:
            if t.name == "End-effector position":
                # t.setK(np.diag([0.3,0.3,0.3]))
                t.setK(np.diag([0.7,0.7,0.7]))
            elif t.name == "Joint position":
                t.setK(np.array([1.0]))
            elif t.name == "End-effector orientation":
                t.setK(np.diag([1.0, 1.0, 1.0]))
            elif t.name == "End-effector configuration":
                t.setK(np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            elif t.name == "Obstacle avoidance":
                t.setK(np.diag([1.0, 1.0]))
            elif t.name == "Joint limit":
                t.setK(np.array([0.1]))
            elif t.name == "Base Orientation":
                t.setK(np.array([1.0]))

        # self.K = np.diag([1, 1,1,1,1,1])

        self.dq = [0.0,0.0,0.0,0.0,0.0,0.0]


        # anon variables
        self.current_pose = [0.0,0.0,0.0]
        self.z_aruco = 0.0 # heigh of robot

        self.t = 0
        self.alpha = 0
        self.last_x = 0.0
        self.last_theta = 0.0

        # Maximum linear velocity control action                   
        self.v_max = 0.15
        # Maximum angular velocity control action               
        self.w_max = 0.6  

        #subscribe to joint positions
        self.joints_sub = rospy.Subscriber('/turtlebot/joint_states', JointState, self.get_joints)

        # Subscriber to groundtruth odom tp get odom
        # self.odom_sub = rospy.Subscriber('kobuki/odom', Odometry, self.get_odom)
        self.odom_sub = rospy.Subscriber('/turtlebot/odom', PoseStamped, self.get_odom_stamped)
        
        # publish v and w
        self.cmd_pub = rospy.Publisher('/lin_ang_velocities', Twist, queue_size=1)

        # publish joint velocities
        self.velocity_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray, queue_size=10)

        # Visualizations
        self.EE_pose_pub = rospy.Publisher('/EE_Pose', PoseStamped, queue_size=10)
        self.desired_pose_pub = rospy.Publisher('/desired_Pose', PoseStamped, queue_size=10)

        # Service to set the goal
        rospy.Service('/set_desired',intervention_getpoint, self.set_desired)
        rospy.Service('/goal_reached', Trigger, self.check_reached)

    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        self.z_aruco = odom.pose.pose.orientation.z
        self.t = self.current_pose[0] - self.last_x
        self.last_x = self.current_pose[0]

        self.alpha = self.current_pose[2] - self.last_theta
        self.last_theta = self.current_pose[2]

        self.odom_received =  True

    def get_odom_stamped(self, odom):
        # print('odom receved')
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.orientation.x, 
                                                              odom.pose.orientation.y,
                                                              odom.pose.orientation.z,
                                                              odom.pose.orientation.w])
        self.current_pose = np.array([odom.pose.position.x, odom.pose.position.y, yaw])
        self.t = self.current_pose[0] - self.last_x
        self.last_x = self.current_pose[0]

        self.alpha = self.current_pose[2] - self.last_theta
        self.last_theta = self.current_pose[2]

        # print('odom_received: ', self.current_pose)
        self.odom_received =  True

    def set_desired(self,msg):
        self.desired_received = True
        # print("called")
        # Access the desired information from the message
        position = msg.pose_msg.pose.position
        orientation = msg.pose_msg.pose.orientation

        # # Pass the received values to the desired file
        before= [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z]
        # a=[(before[0]/100),before[1]/100,before[2]/100,1]
        # print('odom_received: ', self.current_pose)
        # print("camera sees",a)
        # x=np.array(a)
        # print(x.shape)
        # after= transfer_camframe_to_world(self.current_pose[0], self.current_pose[1],self.current_pose[2])@ x
        # after=after.T
        # print("after",after)
        # after= list(after)
        # self.sigma_d=after[0:3]
        # self.sigma_d[3:0]= [0,0,0]
        self.sigma_d=[before[0],before[1],before[2]]
        print(" self.sigma_d", self.sigma_d)
        #-----------------------------------------just a desired in world no cam  
        # print(self.sigma_d)
        for t in self.tasks:
            if t.name == "End-effector position":
                t.setDesired(np.array(self.sigma_d[0:3]).reshape(3,1))
        # self.tasks[0].setDesired(np.array(self.sigma_d[0:3]).reshape(3,1))
        return True

    # Service to check if reached
    def check_reached(self, req):
        # Check if the robot is moving
        if self.goal_reached:
            self.goal_reached = False
            return TriggerResponse(success=True, message='EE successfully reached desired')
        else:
            return TriggerResponse(success=False, message='EE not reached desired yet')

    def controller(self):

        P_i = np.eye(6).astype(float)
        # Initialize output vector (joint velocity)
        dq = np.zeros(6).reshape(6,1)

        for t in self.tasks:
            J_i, sigma_err = t.update(self.robot)
            if t.isActive():
                # print(t.name)
                # print('jacob shape: ', J_i.shape)
                # print('error shape: ', sigma_err.shape)
                # if t.name == "Joint limit":
                #     print("--   JOINT TASK    --")
                #     print('jacob: ', J_i)
                #     print('error: ', sigma_err)



                J_i_bar = J_i @ P_i
                K = t.getK()

                # print(K)
                # print(sigma_err)
                x_dot = K @ sigma_err

                # Accumulate velocity
                dq = dq + WDLS(J_i_bar,self.damping, self.weights) @ (x_dot - J_i @ dq)
    
                # Update null-space projector
                P_i = P_i - pinv(J_i_bar) @ J_i_bar

                # print(dq)

                # publish joint vels
                velocity_msg = Float64MultiArray()
                velocity_msg.data = [dq[0],dq[1],dq[2],dq[3]]
                self.velocity_pub.publish(velocity_msg)

                # publish base vels
                alpha_dot = -dq[4]
                t_dot = dq[5]
                self.__send_command__(t_dot, alpha_dot)

                if t.name == "End-effector position" or t.name == 'End-effector configuration':
                    abs_err= np.sqrt(sigma_err[0]**2+sigma_err[1]**2+sigma_err[2]**2)
                    if abs_err < 0.03:
                        self.goal_reached = True
                        self.desired_received = False
                   

        self.publish_points()


    def get_joints(self,data):   
        joint_pos = np.array(data.position)
        if len(joint_pos) == 4:
            # get joints
            [t1, t2, t3, t4] = joint_pos
            # get base
            [x_base, y_base, theta_base] = self.current_pose
            
            # call update to update the manipulator
            self.robot.update(t1,t2,t3,t4,x_base,y_base,theta_base,self.alpha,self.t)
            
            if self.desired_received:
                self.controller()
            

    def publish_points(self):

        new_pose = PoseStamped()
        new_pose.header.frame_id = 'map'
        new_pose.header.stamp = rospy.Time.now()

        curr_pose = self.robot.get_T()

        new_pose.pose.position.x = curr_pose[0]
        new_pose.pose.position.y = curr_pose[1]
        new_pose.pose.position.z = curr_pose[2]

        quaternion = tf.transformations.quaternion_from_euler(0, 0, curr_pose[-1])
        #type(pose) = geometry_msgs.msg.Pose
        new_pose.pose.orientation.x = quaternion[0]
        new_pose.pose.orientation.y = quaternion[1]
        new_pose.pose.orientation.z = quaternion[2]
        new_pose.pose.orientation.w = quaternion[3]

        self.EE_pose_pub.publish(new_pose)
        
        new_pose2 = PoseStamped()
        new_pose2.header.frame_id = 'map'
        new_pose2.header.stamp = rospy.Time.now()
    
        new_pose2.pose.position.x = self.sigma_d[0]
        new_pose2.pose.position.y = self.sigma_d[1]
        new_pose2.pose.position.z = self.sigma_d[2]

        quaternion = tf.transformations.quaternion_from_euler(0, 0, self.sigma_d[-1])
        #type(pose) = geometry_msgs.msg.Pose
        new_pose2.pose.orientation.x = quaternion[0]
        new_pose2.pose.orientation.y = quaternion[1]
        new_pose2.pose.orientation.z = quaternion[2]
        new_pose2.pose.orientation.w = quaternion[3]


        self.desired_pose_pub.publish(new_pose2)


    def __send_command__(self, v, w):
        '''
        Transform linear and angular velocity (v, w) into a Twist message and publish it
        '''
        # print('v,w: ', v, w)
        cmd = Twist()
        cmd.linear.x = np.clip(v, -self.v_max, self.v_max)
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = np.clip(w, -self.w_max, self.w_max)
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    rospy.init_node('JointController', anonymous=True)
    print('node created')
    try:
        t = JointController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    # rospy.spin()