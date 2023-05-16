import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# x = (l2 * cos(t2) + l3 * sin(t3) + l4) * cos(t1)
# y = (l2 * cos(t2) + l3 * sin(t3) + l4) * sin(t1)
# z = -l2 * sin(t2) + l3 * cos(t3) + l1 - l5

# dx/dt1 = -sin(t1) * (l2 * cos(t2) + l3 * sin(t3) + l4)
# dx/dt2 = -(l2 * sin(t2) - l3 * cos(t3)) * sin(t1)
# dx/dt3 = l3 * sin(t1) * cos(t3)

# dy/dt1 = cos(t1) * (l2 * cos(t2) + l3 * sin(t3) + l4)
# dy/dt2 = -(l2 * sin(t2) - l3 * cos(t3)) * cos(t1)
# dy/dt3 = l3 * cos(t1) * sin(t3)

# dz/dt1 = 0
# dz/dt2 = -l2 * cos(t2) - l3 * sin(t3)
# dz/dt3 = -l3 * sin(t3) * cos(t2)

def kinematics(t1, t2, t3, l1, l2, l3, l4, l5):
    # Compute the position of the end effecto
    x = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    y = math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    z = -(l2 * math.cos(t2) + l3 * math.sin(t3) + l1 - l5)

    # Return the position as a numpy array
    return np.array([x, y, z])

def kinematics_total(theta_base, x_base, y_base,x_arm,y_arm,z_arm):
    x_total=(0.5*math.cos(theta_base)-0.5*math.sin(theta_base))*x_arm + (-0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * y_arm +(0.0367*math.cos(theta_base)+x_base)
    y_total=(0.5*math.cos(theta_base)+0.5*math.sin(theta_base))*x_arm + (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * y_arm +(0.0367*math.sin(theta_base)+y_base)
    z_total=-z_arm-0.198

    return x_total,y_total,z_total

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

def jacobian_total(t1, t2, t3, l1, l2, l3, l4, l5, theta_base,x_base, y_base):
    x_arm,y_arm,z_arm = kinematics(t1, t2, t3, l1, l2, l3, l4, l5)
    #https://www.mdpi.com/2218-6581/6/3/17  paper_reference
    J = np.zeros((3, 6))
    J[0, 0] = (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))*(-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4)) + (-0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * (math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)) 
    J[0, 1] = (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))*(-math.cos(t1) * l2 * math.cos(t2)) + (-0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * (-math.sin(t1) * l2 * math.cos(t2))  
    J[0, 2] = (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))*(- math.cos(t1) * l3 * math.sin(t3)) + (-0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * (- math.sin(t1) * l3 * math.sin(t3)) 
    J[0 ,3] = (-0.5*math.sin(theta_base)-0.5*math.sin(theta_base))*x_arm + (0.5*math.sin(theta_base)-0.5*math.cos(theta_base))\
                         * y_arm +(-0.0367*math.sin(theta_base))
    J[0 ,4] =1
    J[0 ,5] =0
    J[1, 0] =(0.5*math.cos(theta_base)+0.5*math.sin(theta_base))*(-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4)) + (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                          * math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4) 
    J[1,1]=(0.5*math.cos(theta_base)+0.5*math.sin(theta_base))*(-math.cos(t1) * l2 * math.cos(t2)) + (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                         * (-math.sin(t1) * l2 * math.cos(t2))
    J[1,2]=(0.5*math.cos(theta_base)+0.5*math.sin(theta_base))*(- math.cos(t1) * l3 * math.sin(t3)) + (0.5*math.cos(theta_base)-0.5*math.sin(theta_base))\
                          * (- math.sin(t1) * l3 * math.sin(t3))
    J[1,3]=(-0.5*math.sin(theta_base)+0.5*math.sin(theta_base))*x_arm + (-0.5*math.sin(theta_base)-0.5*math.cos(theta_base))\
                         * y_arm +(0.0367*math.cos(theta_base))
    J[1,4]=0
    J[1,5]=1
    J[2,0]=-0.198
    J[2,1]=-( l2 * math.sin(t2))-0.198
    J[2,2]=-(-l3 * math.cos(t3))-0.198
    J[2,3]=0
    J[2,4]=0
    J[2,5]=0
    
    return J
 
 

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



plot_arm_workspace()
