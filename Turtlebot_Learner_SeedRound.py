#!/usr/bin/env python
from __future__ import division, absolute_import


#ROS and rospy imports
import rospy
from geometry_msgs.msg import Twist
from take_photo import TakePhoto
import cv2
import tf
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
import sys
import numpy as np
#TFLearn and Tensorflow imports
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from TurtleDVS.srv import *
from TurtleDVS.msg import *



"""
Returns True if the robot is in the goal state, False otherwise
"""
def in_goal(pos):
    pos_x = pos.pose.position.x
    pos_y = pos.pose.position.y
    #hand-defined "goal area", which is around the other side of the divider
    if pos_x < -1.036 and pos_x > -1.888 and pos_y > 0.449 and pos_y < 1.49:
        return True
    else:
        return False 



"""
Calls QValue service to get Q values given input images.
"""
def get_Q_values_client(img1, img2):
    rospy.wait_for_service('get_Q_values')
    try:
        get_Q_values = rospy.ServiceProxy('get_Q_values', GetQValues)
        response = get_Q_values(img1, img2)
        return response.data
    except:
        print "Service call failed"


"""
Calls Gazebo service to return the position of the robot (in the Gazebo simulator). 
"""
def get_model_state_client(model_name):
    
    rospy.wait_for_service('/gazebo/get_model_state')
    
    
    get_model_state_gazebo = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    
    ret = get_model_state_gazebo(model_name, "world")
    return ret

"""
Tests to see whether the robot is stuck or not, to reset
If moved, reset counter to 0. Otherwise, increment counter.
z coordinate doesn't change. 
"""
def did_move(pos1, pos2, counter):
    diff_x = pos1.pose.position.x - pos2.pose.position.x
    diff_y = pos1.pose.position.y - pos2.pose.position.y
    if diff_x < 0.02 and diff_y < 0.02:
        counter += 1
        return counter
    else:
        return 0

#def find_max(Q_values):


def generate_twist(Q_values):
    move_cmd = Twist()
    
    
    Q_array = np.array([Q_values.action_linear, Q_values.action_angular, Q_values.action_linear_angular]) 
    # Twist is a datatype for velocity
    
    max_Q = np.argmax(Q_array)
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.0

    if max_Q == 0:
        move_cmd.linear.x = 0.1
        print "moving forward"
    if max_Q == 1:
        move_cmd.angular.z = 0.1
        print "turning"
    if max_Q == 2:
        print "moving and turning"
        move_cmd.linear.x = 0.1
        move_cmd.angular.z = 0.1
    return move_cmd


class Run():
    def __init__(self):
        
        rospy.init_node('Run', anonymous=False)
	
        
        rospy.loginfo("To stop TurtleBot CTRL + C")

    
    
        

        #Will get set to True if the robot has reached the correct goal area, or if the robot stops moving after 10 steps
        self.finish = False
    
        list1 = [3.6, 2.2, 1.3]
        list2 = [5,7, 6.2, 7.5]
        #curr_Q = get_Q_values_client(list1, list2)
        #print curr_Q
        

        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        
        file1 = open("reward_file.RL", 'w')
        file2 = open("action_taken_file.RL", 'w')
        file3 = open("current_Q_values.RL", 'w')
        
        tfListener = tf.TransformListener()
	
        r = rospy.Rate(30);

        counter = 0
        move_counter = 0 #checks to see if you haven't moved for n previous iterations
        self.pos = ModelState() #initialize pose to nothing
        self.previous_pos = ModelState()
        self.Q = QValues()
        self.prev_image = [0] * 49152
        self.forward_counter = 0
        self.turn_counter = 0
        self.move_cmd = Twist()
        print "before loop"
        while not self.finish:
            action_taken = "forward"
            #self.cmd_vel.publish(move_cmd)
            camera = TakePhoto()
            try:
                self.pos = get_model_state_client("mobile_base")
            except:
                print "Service call failed."
                continue
            if camera.image_received:
                self.curr_image = camera.image
                action_taken = ""
                rospy.loginfo("Got image!")
                cv2.imwrite("working_images/image_" + str(counter) + ".png", camera.image)
                #cv2.imshow("cam", camera.image)
                #print camera.image.shape
                #testing = cv2.resize(camera.image, (0,0), fx=0.4, fy=0.4)
                #grey = cv2.cvtColor(testing, cv2.COLOR_BGR2GRAY)
                #grey_flat = grey.flatten()
                #grey_list = grey_flat.tolist()
                
                #curr_Q = get_Q_values_client(grey_list, self.prev_image)
                #print "after curr_Q"
                #command = generate_twist(curr_Q) #generate new action based on the current Q values
                if self.forward_counter < 47 and self.turn_counter == 0:
                    self.move_cmd.linear.x = 0.1
                    self.move_cmd.angular.z = 0.0
                    self.forward_counter += 1
                    print "forward: {}".format(self.forward_counter)
                    action_taken = "forward"
                elif self.forward_counter >= 47 and self.turn_counter <= 19:
                    self.move_cmd.linear.x = 0.0
                    self.move_cmd.angular.z = 0.1
                    print "turn: {}".format(self.turn_counter)
                    self.turn_counter += 1
                    action_taken = "turn"
                else:
                    self.move_cmd.linear.x = 0.1
                    self.move_cmd.angular.z = 0.0
                    self.forward_counter += 1
                    print "forward: {}".format(self.forward_counter)
                    action_taken = "forward"
                
                    
                self.cmd_vel.publish(self.move_cmd)
                #self.prev_image = grey_list
                # print grey_flat.shape
                #print grey.shape

                if in_goal(self.pos):
                    self.finish = True
                    file1.write("100\n") #reward of 100 for being in goal state
                    continue
                else:
                    file1.write("0\n")
                file2.write(action_taken)
                file2.write("\n")
                #try:
                 #   self.
                
            #move_counter = did_move(self.pos, self.previous_pos, move_counter)
          
                       
            #print "After move counter = ",move_counter
            #if move_counter > 5:
                #rospy.loginfo("Robot stopped.")
                #self.finish = True
            #self.previous_pos = self.pos    
            counter += 1
            r.sleep()
            if self.finish:
                file1.close()
                file2.close()
                
        
        #Run is done, let's learn the most recent path

    
    
    
 
if __name__ == '__main__':
    try:
        Run()
    except:
        rospy.loginfo("GoForward node terminated.")

