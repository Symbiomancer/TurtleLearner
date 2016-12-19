
from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import sys
import os
import numpy as np
import time
from tflearn.layers.conv import highway_conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization



DVS_THRESH = "3"
GAMMA = .9




def load_previous_Q_values(model_name, images):

	file1 = open("predicted_Q_values_multi.RL", "w")
	# Convolutional network building
	#The network is 192x256 because it is downsampled from 480x640 with a factor .4
	network = input_data(shape=[None, 192, 256, 1])
	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 3, activation='sigmoid')
	network = regression(network, optimizer='adam',
                     loss='mean_square',
                     learning_rate=0.01)

	# Train using classifier
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.load(model_name)
	print "Loading images for previous Q values.."
	for image in images:
		#print image.shape
		image = np.reshape(image, (1, 192, 256, 1))
		#print image.shape
		#print image
		prediction = model.predict(image)
		#print prediction
		file1.write("{} {} {}".format(str(prediction[0][0]), str(prediction[0][1]), str(prediction[0][2])))
	file1.close()
	print "Done with previous Q values."
	 
"""
Calculates the label to train on.  yj = rj if episode terminated, else
rj + gamma * max_a(Q_j+1).
"""
def load_labels(run_folder):
	f_reward = open("runs/" + run_folder + "/reward_file.RL", 'r')
	f_action = open("runs/" + run_folder + "/action_taken_file.RL", 'r')
	f_prev_Q = open("runs/" + run_folder + "/current_Q_values.RL", 'r')
	action_line = f_action.readline()
	reward_line = f_reward.readline()
	Q_line = f_prev_Q.readline()
	labels = []
	while action_line != '' or reward_line != '' or Q_line != '':
		

		strip_action = action_line.split()
		strip_reward = reward_line.split()
		strip_Q = Q_line.split()
		#print strip_action, strip_reward
		print strip_Q


		action_line = f_action.readline()
		reward_line = f_reward.readline()
		Q_line = f_prev_Q.readline()
		reward = float(strip_reward[0])
		argmax = 0
		if reward == 100:
			value = reward
		else:
			if (float(strip_Q[0]) > float(strip_Q[1])) and (float(strip_Q[0]) > float(strip_Q[2])): #a crude argmax function
				argmax = 0
			elif (float(strip_Q[1]) > float(strip_Q[0])) and (float(strip_Q[1]) > float(strip_Q[2])): 
				argmax = 1
			elif (float(strip_Q[2]) > float(strip_Q[1])) and (float(strip_Q[2]) > float(strip_Q[0]))	: 
				argmax = 2
		#print type(argmax)
		#print type(strip_Q[argmax])
		#print type(reward)
		#print type(GAMMA)
		value = reward + (GAMMA * float(strip_Q[argmax])) #Bellman Q-value update rule
		#Generate label set
		if strip_action[0] == "forward":
			labels.append([value, 0.0, 0.0])
		elif strip_action[0] == "turn":	
			labels.append([0.0, value, 0.0])
		else:
			labels.append([0.0, 0.0, value]	)
	print len(labels) 
	f_reward.close()
	f_action.close()
	return labels

	 
"""
Calculates the label to train on.  yj = rj if episode terminated, else
rj + gamma * max_a(Q_j+1).
To be called initially when seeding the function.
"""
def load_labels_seed():
	f_reward = open("reward_file.RL", 'r')
	f_action = open("action_taken_file.RL", 'r')
	action_line = f_action.readline()
	reward_line = f_reward.readline()
	action_line = f_action.readline()
	reward_line = f_reward.readline()
	labels = []
	while action_line != '' or reward_line != '':
		

		strip_action = action_line.split()
		strip_reward = reward_line.split()
		
		#print strip_action, strip_reward


		action_line = f_action.readline()
		reward_line = f_reward.readline()
		#Generate label set
		if strip_action[0] == "forward":
			labels.append([float(strip_reward[0]), 0.0, 0.0])
		elif strip_action[0] == "turn":	
			labels.append([0.0, float(strip_reward[0]), 0.0])
		else:
			labels.append([0.0, 0.0, float(strip_reward[0])])
	print len(labels) 
	f_reward.close()
	f_action.close()
	return labels




def load_and_train_highway_network(model_name, images_x, labels_y):
	images_x = np.reshape(images_x, (-1, 192, 256, 1))

	network = input_data(shape=[None, 192, 256, 1], name='input')
	for i in range(5):
		for j in [3, 2, 1]: 
			network = highway_conv_2d(network, 16, j, activation='elu')
		
		network = max_pool_2d(network, 2)
		network = batch_normalization(network)
    
	network = fully_connected(network, 128, activation='elu')
	network = fully_connected(network, 256, activation='elu')
	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.01,
		loss='categorical_crossentropy', name='target')
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.load(model_name)
	model.fit(images_x, labels_y, n_epoch=40,
          show_metric=True, run_id='Q_values_highway')
	model.save(model_name)




def load_and_train_highway_network_initial(model_name, images_x, labels_y):
	images_x = np.reshape(images_x, (-1, 192, 256, 1))

	network = input_data(shape=[None, 192, 256, 1], name='input')
	for i in range(5):
		for j in [3, 2, 1]: 
			network = highway_conv_2d(network, 16, j, activation='elu')
		
		network = max_pool_2d(network, 2)
		network = batch_normalization(network)
    
	network = fully_connected(network, 128, activation='elu')
	network = fully_connected(network, 256, activation='elu')
	network = fully_connected(network, 3, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.01,
		loss='categorical_crossentropy', name='target')
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit(images_x, labels_y, n_epoch=40,
          show_metric=True, run_id='Q_values_highway')
	model.save(model_name)



def load_and_train_network(model_name, images_x, labels_y):
	images_x = np.reshape(images_x, (-1, 192, 256, 1))
	# Convolutional network building
	#The network is 192x256 because it is downsampled from 480x640 with a factor .4
	network = input_data(shape=[None, 192, 256, 1])
	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 256, activation='relu')
	#network = dropout(network, 0.5)
	network = fully_connected(network, 3, activation='sigmoid')
	network = regression(network, optimizer='adam',
                     loss='mean_square',
                     learning_rate=0.01)

	# Train using classifier
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.load(model_name)
	model.fit(images_x, labels_y, n_epoch=50,
          show_metric=True, run_id='Q_values')
	model.save(model_name)




"""
Loads in the images from the DVS_images folder, 
and returns the 2D structure.
"""
	
def load_images(folder):
	#feature_vec = np.empty((0, 192, 256))
	feature_vec = []
	print folder
	print "Loading DVS images from previous run..."
	path, dirs, files = os.walk("runs/"+ folder +"/DVS_images").next()
	num_images = len(files)
	#print num_images
	for a in xrange(num_images):
		#print "image_DVS_{}_{}.png".format(DVS_THRESH, str(a))
		img = cv2.imread('runs/{}/DVS_images/image_DVS_{}_{}.png'.format(folder, DVS_THRESH, str(a)),0) 
		img = cv2.resize(img, (0,0), fx=0.4, fy=0.4)
		ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        #training_images.append(img)
        #imglist = img.tolist()
		#print "2"
        #time.sleep(10)
        #feature_vec = np.append(feature_vec, [img], axis=0)
		feature_vec.append(img)
		#print img.shape
		#print "1"
	#feature_vec = np.array(feature_vec)   
	#print len(feature_vec)
	#feature_vec = feature_vec.reshape([-1, 192, 256, 1])
	#print feature_vec
	print len(feature_vec)
	#print feature_vec
	print "Done loading previous images."
	return feature_vec




if __name__ == "__main__":
     	   
        images = load_images(str(sys.argv[2])) #argv[1]  = folder name 
        if str(sys.argv[1]) == "--initial":
        	labels = load_labels_seed()
        	load_and_train_highway_network_initial("working_models/model_highway_multi.tfl", images, labels)
        elif str(sys.argv[1]) == "--intermediate":
        	labels = load_labels(str(sys.argv[2])) #arg = folder name
        	load_and_train_highway_network("working_models/model_highway_multi.tfl", images, labels)



        
        #load_previous_Q_values("working_models/modellinear_x_0.1.tfl", images, "linear")
        #load_previous_Q_values("working_models/modelangular_z_0.1.tfl", images, "angular")
        #load_previous_Q_values("working_models/modellinear_angular_0.1.tfl", images, "both")
        #load_previous_Q_values("working_models/modelmulti.tfl", images)
        
        #load_network("modelangular_z_0.1.tfl")
        print "Done"
