import numpy as np
import math
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def transfer_camframe_to_world(x_base, y_base,theta_base):
    world_to_base = np.array([
        [np.cos(theta_base), -np.sin(theta_base), 0, x_base],
        [np.sin(theta_base), np.cos(theta_base), 0, y_base],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    base_to_cam= np.array([
        [0, 0,1,0.122],
        [1,0,0,-0.033],
        [0,1,0,0.082],
        [0, 0, 0, 1]
    ])

    # Perform matrix multiplication
    world_to_cam= np.dot(world_to_base,base_to_cam)


    return world_to_cam

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    # Implement the formula to compute the DLS of matrix A.
    I = np.eye(np.shape(A)[0])
    brak1 = np.linalg.inv(np.dot(A, np.transpose(A)) + damping**2 * I)
    return np.dot(np.transpose(A),brak1)

def rpy_to_quaternion(rpy):

    [roll, pitch, yaw] = rpy

    # Calculate the trigonometric functions
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    # Calculate the quaternion components
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([w, x, y, z])

#_______________________________________________________________FOR THE ARM ONLY _____________________________________

def kinematics(t1, t2, t3, l1, l2, l3, l4, l5):
    # Compute the position of the end effecto
    x = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    y = math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    z = -(l2 * math.cos(t2) + l3 * math.sin(t3) + l1 - l5)
    return np.array([x, y, z])
def jacobian(t1, t2, t3, l1, l2, l3, l4, l5):
    # Compute the Jacobian matrix
    J = np.zeros((3, 3))
    J[0, 0] = -math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4) #dx t1
    J[0, 1] = -math.cos(t1) * l2 * math.cos(t2) #dx t2
    J[0, 2] = - math.cos(t1) * l3 * math.sin(t3) #dx t3
    J[1, 0] = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4) #dy t1
    J[1, 1] = -math.sin(t1) * l2 * math.cos(t2) #dy t2
    J[1, 2] = - math.sin(t1) * l3 * math.sin(t3) #dy d3
    J[2, 0] = 0 #dz t1
    J[2, 1] = l2 * math.sin(t2) #dz t2
    J[2, 2] = -l3 * math.cos(t3) #dz t3
    return J
#____________________________________________________________FOR ALL (ARM AND BASE)_______________________________________________________________

def kinematics_total(t1, t2, t3,t4, l1, l2, l3, l4, l5,x_base, y_base,theta_base,alpha,t):
        #------------------------------------------ just for simplification------------------------------
    a= (math.cos(theta_base)* math.cos(alpha))-(math.sin(theta_base)*math.sin(alpha))
    b= (-math.cos(theta_base)* math.sin(alpha))-(math.sin(theta_base)*math.cos(alpha))
    c= (math.cos(theta_base)* math.cos(alpha)*t)-(math.sin(theta_base)*math.sin(alpha)*t) + x_base
    d= (math.sin(theta_base)* math.cos(alpha))+(math.cos(theta_base)*math.sin(alpha))
    e= (-math.sin(theta_base)* math.sin(alpha))+(math.cos(theta_base)*math.cos(alpha))
    f= (math.sin(theta_base)* math.cos(alpha)*t)+(math.cos(theta_base)*math.sin(alpha)*t) + y_base
    #--------------------------------------------------------------------------------------------------
    
    x_arm = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    y_arm = math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    z_arm= -(l2 * math.cos(t2) + l3 * math.sin(t3) + l1 - l5)

    x_total=-b*x_arm + a*y_arm + c + 0.05*a
    y_total=-e*x_arm + d*y_arm + f + 0.05*d
    z_total= z_arm-0.198
    row_total=0
    pitch_total=0
    yaw_total= t4+theta_base+t1

    return x_total,y_total,z_total,row_total,pitch_total,yaw_total

def jacobian_total(t1, t2, t3, l1, l2, l3, l4, l5,x_base, y_base,theta_base,alpha,t):
    x_arm,y_arm,z_arm = kinematics(t1, t2, t3, l1, l2, l3, l4, l5)
        #------------------------------------------ just for simplification------------------------------
    a= (math.cos(theta_base)* math.cos(alpha))-(math.sin(theta_base)*math.sin(alpha))
    b= (-math.cos(theta_base)* math.sin(alpha))-(math.sin(theta_base)*math.cos(alpha))
    c= (math.cos(theta_base)* math.cos(alpha)*t)-(math.sin(theta_base)*math.sin(alpha)*t) + x_base
    d= (math.sin(theta_base)* math.cos(alpha))+(math.cos(theta_base)*math.sin(alpha))
    e= (-math.sin(theta_base)* math.sin(alpha))+(math.cos(theta_base)*math.cos(alpha))
    f= (math.sin(theta_base)* math.cos(alpha)*t)+(math.cos(theta_base)*math.sin(alpha)*t) + y_base

    #---------------for simplification d/d_alpha---------------
    A=-math.cos(theta_base)* math.sin(alpha) - math.sin(theta_base)*math.cos(alpha)
    B=- math.cos(theta_base)* math.cos(alpha)- math.sin(theta_base)*math.sin(alpha)
    C=-math.cos(theta_base)* math.sin(alpha)*t - math.sin(theta_base)*math.cos(alpha)*t
    D= math.cos(theta_base)* math.cos(alpha)-math.sin(theta_base)*math.sin(alpha)
    E= -(math.sin(theta_base)* math.cos(alpha))-(math.cos(theta_base)*math.sin(alpha))
    F=-math.sin(theta_base)* math.sin(alpha)*t + math.cos(theta_base)*math.cos(alpha)*t

    J = np.zeros((6, 6))

    # #--------------------------------------x all-----------------------------------------
    J[0, 0] = -b*(-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4)) + a*( math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)) 

    J[0, 1] = -b*(-math.cos(t1) * l2 * math.cos(t2)) + a*(-math.sin(t1) * l2 * math.cos(t2) ) 

    J[0, 2] = -b*(- math.cos(t1) * l3 * math.sin(t3)) + a*(- math.sin(t1) * l3 * math.sin(t3)) 
    J[0,3] = 0

    J[0 ,4] = -B * x_arm + A*y_arm + C + 0.05*A
    J[0 ,5] =math.cos(theta_base)* math.cos(alpha)-math.sin(theta_base)*math.sin(alpha)
    
#     #--------------------------------------y all-----------------------------------------
    J[1, 0] = -e*((-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4))) + d*( math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4))
    # e*x_arm + d*y_arm + f + 0.05*d

    J[1,1]= e*(-math.cos(t1) * l2 * math.cos(t2)) + d*(-math.sin(t1) * l2 * math.cos(t2))

    J[1,2]=e*(- math.cos(t1) * l3 * math.sin(t3)) + d*(- math.sin(t1) * l3 * math.sin(t3)) 
    J[1,3]=0

    J[1,4]= E*x_arm + D*y_arm + F + 0.05*D

    J[1,5]=math.sin(theta_base)* math.cos(alpha)+math.cos(theta_base)*math.sin(alpha)

#     #--------------------------------------z all-----------------------------------------
    J[2, 0] = 0 #dz t1
    J[2, 1] = l2 * math.sin(t2) #dz t2
    J[2, 2] = -l3 * math.cos(t3) #dz t3
    J[2,3]=0
    J[2,4]=0
    J[2,5]=0
    
    #--------------------------------------row and pitch-----------------------------------------
    J[3,:]=0 
    J[4,:]=0
    #-------------------------------------- yaw-----------------------------------------
    J[5,0]=1
    J[5,1]=0
    J[5,2]=0
    J[5,3]=1
    J[5,4]=1
    J[5,5]=0
    return J 

def transfer_camframe_to_world(x_base, y_base,theta_base):
    TCW = np.zeros((4, 4))
    TCW[0,0]= -math.sin(theta_base)
    TCW[0,1]= 0
    TCW[0,2]= math.cos(theta_base)
    TCW[0,3]= 0.122*math.cos(theta_base) + 0.033*math.sin(theta_base) + x_base
    TCW[1,0]= math.cos(theta_base)
    TCW[1,1]= 0
    TCW[1,2]= math.sin(theta_base)
    TCW[1,3]= 0.122*math.sin(theta_base) - 0.033*math.cos(theta_base) + y_base
    TCW[2,0]= 0
    TCW[2,1]= -1
    TCW[2,2]= 0
    TCW[2,3]= -0.082
    TCW[3,0]= 0
    TCW[3,1]= 0
    TCW[3,2]= 0
    TCW[3,3]= 1

    return TCW




def plot_arm_workspace():
    l1 = 0.108
    l2 = 0.142
    l3 = 0.1588
    l4 = 0.0565
    l5 = 0.0722

    t1_range = np.arange(-1.2, 1.2, 0.1)
    t2_range = np.arange(-1.2, -0.3, 0.1)
    t3_range = np.arange(-1.2, -0.3, 0.1)
    x_vals = []
    y_vals = []
    z_vals = []

    for t1 in t1_range:
        for t2 in t2_range:
            for t3 in t3_range:
                x = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
                y = math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
                z = -(l2 * math.cos(t2) + l3 * math.sin(t3) + l1 - l5)

                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, s=1, c=z_vals, cmap='viridis')

    # randomly select a point and color it red
    index = random.randint(0, len(x_vals)-1)
    ax.scatter(x_vals[index], y_vals[index], z_vals[index], s=50, c='r')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    return (x_vals[index], y_vals[index], z_vals[index])



# plot_arm_workspace()
