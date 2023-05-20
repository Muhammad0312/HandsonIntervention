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
from functions_robotics import *
from std_msgs.msg import Float64MultiArray

class JointController:
    def __init__(self):

        self.desired = [1.0,1.0,-0.2,0,0,1.57]

        self.l1 = 0.108
        self.l2 = 0.142
        self.l3 = 0.1588
        self.l4 = 0.0565
        self.l5 = 0.0722
        self.K = np.diag([1, 1,1,1,1,1])
        self.orient = 0
        self.current_pose = [0.0,0.0,0.0]

        self.t = 0
        self.alpha = 0

        self.last_x = 0.0
        self.last_theta = 0.0

        # Maximum linear velocity control action                   
        self.v_max = 0.15
        # Maximum angular velocity control action               
        self.w_max = 0.3  

        self.joints_sub = rospy.Subscriber('/turtlebot/joint_states', JointState, self.get_joints)

        # Subscriber to groundtruth
        self.odom_sub = rospy.Subscriber('/turtlebot/stonefish_simulator/ground_truth_odometry', Odometry, self.get_odom)
        #Publisher Linear and angular veocities to the topic which will convert to joint velocities and publish to wheel_velocities topic
        self.cmd_pub = rospy.Publisher('/lin_ang_velocities', Twist, queue_size=1)

        self.velocity_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray, queue_size=10)

        self.EE_pose_pub = rospy.Publisher('/EE_Pose', PoseStamped, queue_size=10)
        self.desired_pose_pub = rospy.Publisher('/desired_Pose', PoseStamped, queue_size=10)


    def controller(self,J, last_T,desired):
        
        sigma_d=  np.array(desired) 
        
        sigma = np.array(last_T)

        sigma_err = sigma_d - sigma
        
        # print('sigma_err: ', sigma_err)

        tracking_err = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        err = tracking_err + np.dot(self.K, sigma_err)

        # print('err: ', sigma_err)
        
        damping_factor = 0.01
        J_t_J = np.dot(J.transpose(), J)
        J_t_J_damped = J_t_J + damping_factor * np.eye(6) # add damping to diagonal
        J_t_J_inv = np.linalg.inv(J_t_J_damped)
        J_t_J_inv_J_t = np.dot(J_t_J_inv, J.transpose())
        
        dq = np.dot(J_t_J_inv_J_t, err)

        # print(dq)

        # # Publish the velocity command
        velocity_msg = Float64MultiArray()
        velocity_msg.data = [dq[0],dq[1],dq[2],dq[3]]
        self.velocity_pub.publish(velocity_msg)
        # print("Joint published:", velocity_msg.data)

        alpha_dot = dq[4]
        t_dot = dq[5]
        self.__send_command__(t_dot, alpha_dot)

        # self.publish_points(sigma,sigma_d)

        # return dq


    def get_joints(self,data):   
        joint_pos = np.array(data.position)
        if len(joint_pos) == 4:

            # get joints
            [t1, t2, t3, t4] = joint_pos
            
            # get base
            [x_base, y_base, theta_base] = self.current_pose
            
            # compute kinematics
            [x_total, y_total, z_total, roll, pitch, yaw] = \
            kinematics_total(t1, t2, t3,t4, self.l1, self.l2, self.l3, self.l4, self.l5,\
                             x_base, y_base,theta_base,self.alpha,self.t)
         
            last_T = [x_total, y_total, z_total, roll, pitch, yaw]

            # compute jacobian 
            J = jacobian_total(t1, t2, t3, self.l1, self.l2, self.l3, self.l4, self.l5\
                               ,x_base, y_base,theta_base,self.alpha,self.t)
         
            self.publish_points([x_total,y_total,z_total,yaw],np.array(self.desired))
            
            self.controller (J, last_T,self.desired)
            



    def publish_points(self,sigma,sigma_d):

        new_pose = PoseStamped()
        new_pose.header.frame_id = 'world_ned'
        new_pose.header.stamp = rospy.Time.now()

        new_pose.pose.position.x = sigma[0]
        new_pose.pose.position.y = sigma[1]
        new_pose.pose.position.z = sigma[2]

        quaternion = tf.transformations.quaternion_from_euler(0, 0, sigma[3])
        #type(pose) = geometry_msgs.msg.Pose
        new_pose.pose.orientation.x = quaternion[0]
        new_pose.pose.orientation.y = quaternion[1]
        new_pose.pose.orientation.z = quaternion[2]
        new_pose.pose.orientation.w = quaternion[3]

        self.EE_pose_pub.publish(new_pose)
        
        new_pose2 = PoseStamped()
        new_pose2.header.frame_id = 'world_ned'
        new_pose2.header.stamp = rospy.Time.now()
    
        new_pose2.pose.position.x = sigma_d[0]
        new_pose2.pose.position.y = sigma_d[1]
        new_pose2.pose.position.z = sigma_d[2]
        self.desired_pose_pub.publish(new_pose2)

    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                            odom.pose.pose.orientation.y,
                                                            odom.pose.pose.orientation.z,
                                                            odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        
        self.t = self.current_pose[0] - self.last_x
        self.last_x = self.current_pose[0]

        self.alpha = self.current_pose[2] - self.last_theta
        self.last_theta = self.current_pose[2]

        # print('odom_received: ', self.current_pose)
        self.odom_received =  True

    def __send_command__(self, v, w):
        '''
        Transform linear and angular velocity (v, w) into a Twist message and publish it
        '''
        print('v,w: ', v, w)
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