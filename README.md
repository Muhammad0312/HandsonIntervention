# Hands-on Intervention

last update:
5May 9.04: Package made, orientation of EE set

## HOW TO RUN:

**launch sim and code sperately**
1. roslaunch turtlebot_simulation swiftpro_basic.launch
2. rosrun hands-on-intervention controller.py
3. rosrun hands-on-intervention FK_diff_drive.py

**Run all**
roslaunch hands-on-intervention hands-on-intervention.launch

If you dont see markers on rviz, add them, topics are EE_pose and desired_pose
save rviz so next time you dont have to add