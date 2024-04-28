# Hands-on Intervention

last update:
30 May 

# Results

![Simulation](media/intervention_perception.gif)
![RealRobot](media/intervention.gif)

# Intervention

- roslaunch autonomous_task_NAK intervention_stonefish.launch
- roslaunch autonomous_task_NAK intervention_hardware.launch


## Dev Phase: (Ignore)

**launch mobile stuff**
1. roslaunch turtlebot_simulation turtlebot_hoi.launch
2. rosrun pose-graph-slam integration.py
3. rosrun hands-on-intervention FK_diff_drive.py
4. rosrun hands-on-intervention mobile_controller.py

**launch only arm**
1. roslaunch turtlebot_simulation swiftpro_basic.launch
2. rosrun hands-on-intervention controller.py
3. rosrun hands-on-intervention FK_diff_drive.py


**launch only arm**
1. roslaunch turtlebot_simulation swiftpro_basic.launch
2. rosrun hands-on-intervention controller.py
3. rosrun hands-on-intervention FK_diff_drive.py

**Run all**
roslaunch hands-on-intervention hands-on-intervention.launch

If you dont see markers on rviz, add them, topics are EE_pose and desired_pose
save rviz so next time you dont have to add

**integration with perception**
- roslaunch turtlebot_simulation turtlebot_hoi.launch
- rosrun hands-on-intervention FK_diff_drive.py
- rosrun pose-graph-slam integration.py 
- rosrun hands_on_perception aruco_pose_realsense_stonefish.py
- rosrun hands-on-intervention mobile_controller.py
