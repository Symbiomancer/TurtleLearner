#!/bin/bash

for i in `seq 3 10`;
do
    rosservice call gazebo/reset_world
    wait
    rosrun TurtleDVS Turtlebot_Learner.py 
    wait
    if [[ $? = 0 ]]; then
	echo "robot finished successfully"
	mkdir runs/run$i
	mkdir runs/run$i/RGB_images
	mkdir runs/run$i/DVS_images
	mv *.RL runs/run$i/
	mv image_* runs/run$i/RGB_images/
	wait
	echo "creating DVS files..."
	python generate_DVS.py runs/run$i 3
	wait
	echo "finished creating DVS files."
    else
	echo "robot must have crashed...restarting"
	source clean.sh
    fi
done

