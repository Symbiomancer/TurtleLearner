#!/usr/bin/env python


import rospy
from geometry_msgs.msg import Twist
from take_photo import TakePhoto
import cv2
import tf
#import roslib; roslib.load_manifest('gazebo')
from gazebo_msgs.srv import GetModelState

def get_model_state_client(model_name):
    print "in get model state ", model_name
    rospy.wait_for_service('/gazebo/get_model_state')
    print "after"
    
    get_model_state_gazebo = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    print "after serviceproxy"
    ret = get_model_state_gazebo(model_name, "world")
    return ret
        


class Run():
    def __init__(self):
        # initiliaze
        rospy.init_node('Run', anonymous=False)
	
        # tell user how to stop TurtleBot
	rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)

    
       
        

	# Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        
        

        tfListener = tf.TransformListener()
	
        r = rospy.Rate(10);

        # Twist is a datatype for velocity
        move_cmd = Twist()
	# let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.1
	# let's turn at 0 radians/s
	move_cmd.angular.z = 0
        counter = 0
    
	# as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
	    # publish the velocity
            self.cmd_vel.publish(move_cmd)
	    # wait for 0.1 seconds (10 HZ) and publish again
            camera = TakePhoto()
            try:
                tfListener.waitForTransform("/odom", "/base_footprint", rospy.Time(0), rospy.Duration(4.0))
                (position, orientation) = tfListener.lookupTransform("/odom", "/base_footprint", rospy.Time(0))
                print "position: ", position
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "error with transform"
                continue
            try:
                print get_model_state_client("mobile_base")
            except:
                print "Service call failed."
                continue
            if camera.image_received:
                rospy.loginfo("Got image!")
                #cv2.imwrite("image_" + str(counter) + ".png", camera.image)
                #cv2.imshow("cam", camera.image)
                
            counter += 1
            r.sleep()
    
    def gazebo_callback(data):
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.Pose.position)
        print data.Pose.position
        
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        
	# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())

        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)


 
if __name__ == '__main__':
    try:
        Run()
    except:
        rospy.loginfo("GoForward node terminated.")

