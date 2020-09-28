import math
import re
import inspect
import os
import numpy as np
import tensorflow as tf
import time
import random
import cv2
import tensorflow.contrib.slim as slim
import tensorflow as tf
one_mlabel = np.array([1,1,1,1,1,  1,1,1,1,1,  1,1,1,1,   1,1,1,1,1  ,0,0,0,0,0])
one_clabel = np.array([1,1,1,1,1,  1,1,1,1,1,  1,1,1,1,   0,0,0,0,0  ,0,0,0,0,0])
tm_labels = []
tc_labels = []
test_images = []
test_labels = []
index1 = 0
index = 0

def load_test_list():
	global test_images
	global test_labels;global tm_labels;global tc_labels
    	directory = './test/test1/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   0,0,0,1         ,0,0,0,0,1          ,0,0,0,0,1]);
    
    	directory = './test/test2/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   0,0,0,1         ,0,0,0,0,1          ,0,0,0,1,0]);
    
    	directory = './test/test3/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   0,0,0,1         ,0,0,0,0,1          ,0,0,1,0,0]);
    
    	directory = './test/test4/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   0,0,1,0         ,0,0,0,0,1          ,0,0,0,0,1]);
    
    	directory = './test/test7/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   0,0,1,0         ,0,0,0,1,0          ,0,0,0,1,0]);

    	directory = './test/test9/';
    	for filename in [y for y in os.listdir(directory)]:

    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,0,   1,0,0,0         ,0,0,0,0,1          ,0,0,0,1,0])
    
    	directory = './test/test10/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([1,0,0,0,0    ,0,0,0,0,0,   0,0,0,0         ,0,0,0,0,1          ,0,0,0,0,1]);
    
    	directory = './test/test14/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([1,0,0,0,0    ,0,0,0,0,0,   0,0,0,0         ,0,0,0,0,1          ,0,1,0,0,0]);
    
    	directory = './test/test15/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([1,0,0,0,0    ,0,0,0,0,0,   0,0,0,0         ,0,0,0,0,1          ,1,0,0,0,0]);
    
    	directory = './test/test16/';
    	for filename in [y for y in os.listdir(directory)]:
    
    				test_images.append(directory+filename)
    				test_labels.append([0,0,0,0,0    ,0,0,0,0,1,   0,0,0,0         ,0,0,0,0,1          ,0,0,0,0,1]);

	zipped = zip(test_images,test_labels)
	random.shuffle(zipped)
	test_images,test_labels = zip(*zipped);print('load down')
	return 	len(test_images)


def get_test(batch_size):
	"only choose the top-left blocks (8*48)*(8*48)*3"
	global index1;global test_images;global test_labels
	img_size = 48
	Max_index = len(test_images);
	index1 = index1%Max_index
	ori_img = cv2.imread(test_images[index1])
	height = int(math.sqrt(batch_size))
	imgs = []
	labels = []
	for i in range(height):
		for j in range(height):
				img = ori_img[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size,:].reshape([3,48,48])
				imgs.append(img)
				labels.append(test_labels[index1])
	tm_label = labels&one_mlabel
	tc_label = labels&one_clabel
	index1 = (index1+1)%Max_index
	return imgs,labels,tm_label,tc_label




