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
import signal

from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerRequest, EmptyResponse, TriggerResponse
import matplotlib.pyplot as plt

class JointController:
    def __init__(self):

        self.sigma_d = [0.0,0.0,0.0,0,0,0]
        

        self.desired_received = False
        self.goal_reached = False

        self.damping = 0.1
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
        self.tasks = [JointLimit("Joint limit", -1.57, 1.57, 0.5, 0),
                      JointLimit("Joint limit", -1.57, 0, 0.2, 1),
                      JointLimit("Joint limit", -1.57, 0, 0.2, 2),
                      JointLimit("Joint limit", -1.57, 1.57, 0.5, 3),
                      Position2D("End-effector position", np.array([0.2,0.2,0.5]).reshape(3,1))]
        
        # self.tasks = [Position2D("End-effector position", np.array([0.2,0.2,0.1]).reshape(3,1))]
                      
                      

        # Set value of K for all tasks
        for t in self.tasks:
            if t.name == "End-effector position":
                t.setK(np.diag([0.5,0.5,0.5]))
            elif t.name == "Joint position":
                t.setK(np.array([1.0]))
            elif t.name == "End-effector orientation":
                t.setK(np.diag([1.0, 1.0, 1.0]))
            elif t.name == "End-effector configuration":
                t.setK(np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            elif t.name == "Obstacle avoidance":
                t.setK(np.diag([1.0, 1.0]))
            elif t.name == "Joint limit":
                t.setK(np.array([1.0]))
            elif t.name == "Base Orientation":
                t.setK(np.array([1.0]))

        # self.K = np.diag([1, 1,1,1,1,1])

        self.dq = [0.0,0.0,0.0,0.0,0.0,0.0]


        # anon variables
        self.current_pose = [0.0, 0.0, 0.0]

        self.t = 0
        self.alpha = 0
        self.last_x = 0.0
        self.last_theta = 0.0

        # Maximum linear velocity control action                   
        self.v_max = 0.15
        # Maximum angular velocity control action               
        self.w_max = 0.3  

        self.error = []
        # self.start_time = time.time()
        self.time = []
        self.joints = []


        #subscribe to joint positions
        self.joints_sub = rospy.Subscriber('/turtlebot/joint_states', JointState, self.get_joints)

        # Subscriber to groundtruth odom tp get odom
        self.odom_sub = rospy.Subscriber('/turtlebot/stonefish_simulator/ground_truth_odometry', Odometry, self.get_odom)
        
        # publish v and w
        self.cmd_pub = rospy.Publisher('/lin_ang_velocities', Twist, queue_size=1)

        # publish joint velocities
        self.velocity_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray, queue_size=10)

        # Visualizations
        self.EE_pose_pub = rospy.Publisher('/EE_Pose', PoseStamped, queue_size=10)
        self.desired_pose_pub = rospy.Publisher('/desired_Pose', PoseStamped, queue_size=10)

        # Service to set the goal
        rospy.Service('/set_desired', intervention_getpoint, self.set_desired)
        rospy.Service('/goal_reached', Trigger, self.check_reached)

    def keyboard_interrupt_handler(self, signal, frame):
        time_arr = np.array(self.time)
        time_arr = time_arr - time_arr[0]
        error_arr = np.array(self.error)
        joints_arr = np.array(self.joints)
        q1_min = self.tasks[0].q_min
        q1_max = self.tasks[0].q_max
        q2_min = self.tasks[1].q_min
        q2_max = self.tasks[1].q_max
        q3_min = self.tasks[2].q_min
        q3_max = self.tasks[2].q_max
        q4_min = self.tasks[3].q_min
        q4_max = self.tasks[3].q_max
        print('Time array: ', time_arr.shape)
        print('Error array: ', error_arr.shape)

        fig, ax = plt.subplots(figsize=(15,10))

        # Plot data  
        ax.plot(time_arr, error_arr, label='norm error')
        ax.plot(time_arr, joints_arr[:, 0], label='q1')
        ax.plot(time_arr, joints_arr[:, 1], label='q2')
        ax.plot(time_arr, joints_arr[:, 2], label='q3')
        ax.plot(time_arr, joints_arr[:, 3], label='q4')
        ax.plot(time_arr, q1_min*np.ones(time_arr.shape), label='q1 min limit', linestyle = 'dashed')
        ax.plot(time_arr, q1_max*np.ones(time_arr.shape), label='q1 max limit', linestyle = 'dashed')
        ax.plot(time_arr, q2_min*np.ones(time_arr.shape), label='q2 min limit', linestyle = 'dashed')
        ax.plot(time_arr, q2_max*np.ones(time_arr.shape), label='q2 max limit', linestyle = 'dashed')
        ax.plot(time_arr, q3_min*np.ones(time_arr.shape), label='q3 min limit', linestyle = 'dashed')
        ax.plot(time_arr, q3_max*np.ones(time_arr.shape), label='q3 max limit', linestyle = 'dashed')
        ax.plot(time_arr, q4_min*np.ones(time_arr.shape), label='q4 min limit', linestyle = 'dashed')
        ax.plot(time_arr, q4_max*np.ones(time_arr.shape), label='q4 max limit', linestyle = 'dashed')

        ax.set_xlabel('Time[s]')
        ax.set_ylabel('Error [m]') 
            
        # Add title and legend
        ax.grid() 
        ax.set_title('Joint Positions')
        ax.legend()
        plt.savefig('/home/mawais/images/image'+str(np.round(time.time(), 2))+'.png')
        plt.close()
    # Add your desired code here

    def set_desired(self,msg):
        self.desired_received = True
        # print("called")
        # Access the desired information from the message
        position = msg.pose_msg.pose.position
        orientation = msg.pose_msg.pose.orientation

        # Pass the received values to the desired file
        before= [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z]

        #-----------------------------------------just a desired in world no cam 
        self.sigma_d = before[0:3]
        self.sigma_d[3:0]= [0, 0, 0] 
        # print(self.sigma_d)
        for t in self.tasks:
            if t.name == "End-effector position":
                t.setDesired(np.array(self.sigma_d[0:3]).reshape(3,1))
        return True

    # Service to check if reached
    def check_reached(self, req):
        # print('shitititit')
        # Check if the robot is moving
        if self.goal_reached:
            self.goal_reached = False
            # plotting joint position over time

            # self.time = []
            # self.error = []

            # Show the plot
            # plt.show()  
            return TriggerResponse(success=True, message='EE successfully reached desired')
        else:
            return TriggerResponse(success=False, message='EE not reached desired yet')

    def controller(self):

        P_i = np.eye(6).astype(float)
        # Initialize output vector (joint velocity)
        dq = np.zeros(6).reshape(6,1)

        for t in self.tasks:
            J_i, sigma_err = t.update(self.robot)
            # print(t.name, ' : ', sigma_err)
            # self.time.append(time.time())
            # self.error.append(sigma_err)
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
                # dq = dq + DLS(J_i_bar,self.damping) @ (x_dot - J_i @ dq) 
                dq = dq + WDLS(J_i_bar,self.damping, self.weights) @ (x_dot - J_i @ dq)
    
                # Update null-space projector
                P_i = P_i - pinv(J_i_bar) @ J_i_bar

                # print(dq)

                # publish joint vels
                velocity_msg = Float64MultiArray()
                velocity_msg.data = [dq[0],dq[1],dq[2],dq[3]]
                self.velocity_pub.publish(velocity_msg)

                # publish base vels
                alpha_dot = dq[4]
                t_dot = dq[5]
                self.__send_command__(t_dot, alpha_dot)

                if t.name == "End-effector position" or t.name == 'End-effector configuration':
                    abs_err= np.sqrt(sigma_err[0]**2+sigma_err[1]**2+sigma_err[2]**2)
                    self.joints.append([self.robot.q1, self.robot.q2, self.robot.q3, self.robot.q4])
                    self.time.append(time.time())
                    self.error.append(abs_err[0])
                    # print(abs_err)
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
        new_pose.header.frame_id = 'world_ned'
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
        new_pose2.header.frame_id = 'world_ned'
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
        signal.signal(signal.SIGINT, t.keyboard_interrupt_handler)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    # rospy.spin()