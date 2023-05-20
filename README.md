# Hands-on Intervention

last update:
5May 9.04: Package made, orientation of EE set

## HOW TO RUN:

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