import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def kinematics(t1, t2, t3, l1, l2, l3, l4, l5):
    # Compute the position of the end effecto
    x = math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    y = math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)
    z = -(l2 * math.cos(t2) + l3 * math.sin(t3) + l1 - l5)

    # Return the position as a numpy array
    return np.array([x, y, z])

def kinematics_total(x_arm,y_arm,z_arm, x_base, y_base,theta_base,alpha,t):
    #------------------------------------------ just for simplification------------------------------
    a= (math.cos(theta_base)* math.cos(alpha))-(math.sin(theta_base)*math.sin(alpha))
    b= (-math.cos(theta_base)* math.sin(alpha))-(math.sin(theta_base)*math.cos(alpha))
    c= (math.cos(theta_base)* math.cos(alpha)*t)-(math.sin(theta_base)*math.sin(alpha)*t) + x_base
    d= (math.sin(theta_base)* math.cos(alpha))+(math.cos(theta_base)*math.sin(alpha))
    e= (-math.sin(theta_base)* math.sin(alpha))+(math.cos(theta_base)*math.cos(alpha))
    f= (math.sin(theta_base)* math.cos(alpha)*t)+(math.cos(theta_base)*math.sin(alpha)*t) + y_base

    x_total=-b*x_arm + a*y_arm + c + 0.05*a
    y_total=-e*x_arm + d*y_arm + f + 0.05*d
    z_total= z_arm-0.198

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

    J = np.zeros((3, 5))

    # #--------------------------------------x all-----------------------------------------
    J[0, 0] = -b*(-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4)) + a*( math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4)) 

    J[0, 1] = -b*(-math.cos(t1) * l2 * math.cos(t2)) + a*(-math.sin(t1) * l2 * math.cos(t2) ) 

    J[0, 2] = -b*(- math.cos(t1) * l3 * math.sin(t3)) + a*(- math.sin(t1) * l3 * math.sin(t3)) 

    J[0 ,3] = -B * x_arm + A*y_arm + C + 0.05*A
    J[0 ,4] =math.cos(theta_base)* math.cos(alpha)-math.sin(theta_base)*math.sin(alpha)
    
#     #--------------------------------------y all-----------------------------------------
    J[1, 0] = -e*((-math.sin(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3)+l4))) + d*( math.cos(t1) * (-l2 * math.sin(t2) + l3 * math.cos(t3) + l4))
    # e*x_arm + d*y_arm + f + 0.05*d

    J[1,1]= e*(-math.cos(t1) * l2 * math.cos(t2)) + d*(-math.sin(t1) * l2 * math.cos(t2))

    J[1,2]=e*(- math.cos(t1) * l3 * math.sin(t3)) + d*(- math.sin(t1) * l3 * math.sin(t3)) 

    J[1,3]= E*x_arm + D*y_arm + F + 0.05*D

    J[1,4]=math.sin(theta_base)* math.cos(alpha)+math.cos(theta_base)*math.sin(alpha)

#     #--------------------------------------z all-----------------------------------------
    J[2, 0] = 0 #dz t1
    J[2, 1] = l2 * math.sin(t2) #dz t2
    J[2, 2] = -l3 * math.cos(t3) #dz t3
    J[2,3]=0
    J[2,4]=0
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



# plot_arm_workspace()
