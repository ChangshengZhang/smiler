#!/usr/bin/python
# File Name: Analysis.py
# Author: Changsheng Zhang
# mail: zhangcsxx@gmail.com
# Created Time: Fri 29 Apr 2016 07:21:08 PM CST

#########################################################################

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)

def handleFeatures(file_path,start_size_num):
	
	file_name_list = os.listdir(file_path)
	
	label = {}
	features_data = []
	label_data = []
	
	for file_name in file_name_list:		

		for lines in open(file_path+file_name):
			features = []
			line = lines.strip("\n").split(",")

			if line[0] == "unknown":
				continue

			for ii in xrange(len(line)):
				if ii == 0:
					if label.has_key(line[ii]):
						label_data.append(label[line[ii]])
					else:
						label[line[ii]] = len(label)
						label_data.append(label[line[ii]])
						
				elif ii == 1 or ii == 3 or ii ==5: 
					continue
				#src port, des port, etc
				elif ii == 2 or ii ==4 or (ii >=6 and ii <=12):
					features.append(int(line[ii]))

				elif ii>=13+start_size_num:
					features.append(float(line[ii]))
				else:
					
					features.append(float(line[ii][0:len(line[ii])-1]))
					if line[ii][-1] == "+":
						features.append(1)
					else:
						features.append(0)

			features_data.append(np.array(features))

	label_data = np.array(label_data)

	features_data = np.array(features_data)

	index = (label_data == -1)

	retain_index = []
	

	#print label
	for ii in range(max(label_data)+1):
		count = sum(label_data ==ii)
		#print count
		if count >1000:
			index[label_data ==ii] = True
			retain_index.append(ii)
		else:
			for item in label.items():
				if item[1] == ii:
					temp = label.pop(item[0])

	label_data = label_data[index]

	for ii in xrange(len(retain_index)):
		label_data[label_data == retain_index[ii]] = ii

	new_label = {}

	for ii in xrange(len(retain_index)):
		for item in label.items():
			if item[1] == retain_index[ii]:
				new_label[item[0]] = ii

	return label_data,Normal(features_data[index]),new_label

	#return label_data,features_data

def Normal(features):

	for ii in xrange(len(features[0])):

		max_num = max(features[:,ii])
		min_num = min(features[:,ii])

		if max_num != min_num:
			features[:,ii] = (features[:,ii]-min_num)*1.0/(max_num-min_num)

	return features


def MLP(data_path,start_size_num):

	labels,features,label_dict = handleFeatures(data_path,start_size_num)
	
	label_length = max(labels)+1
	
	print "label_length",label_length

	features_length = len(features[0])

	train_index = (labels ==-1)
	valid_index = (labels ==-1)
	
	for ii in xrange(len(labels)):
		if ii%5 ==0:
			valid_index[ii] = True
		else:
			train_index[ii] = True
	print sum(train_index),sum(valid_index)

	train_X = features[train_index]
	train_Y = labels[train_index]
	valid_X = features[valid_index]
	valid_Y = labels[valid_index]

	train_label = to_categorical(train_Y,label_length)
	valid_label = to_categorical(valid_Y,label_length)

	model =  Sequential()
	model.add(Dense(100,input_dim = features_length))
	model.add(Activation('tanh'))

	model.add(Dense(40))
	model.add(Activation('tanh'))

	model.add(Dense(label_length))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
										
	model.fit(train_X, train_label, nb_epoch=20, batch_size=32,validation_data=(valid_X,valid_label))

	label_predict = model.predict_classes(valid_X)
	
	accuracy = accuracy_score(valid_Y, label_predict)
	recall = recall_score(valid_Y, label_predict,average = 'micro')
	precision = precision_score(valid_Y, label_predict,average = 'micro')
	f1 = f1_score(valid_Y,label_predict,average = 'micro')

	print('Accuracy: {}'.format(accuracy))
	print('Recall: {}'.format(recall))
	print('Precision: {}'.format(precision))
	print('F1: {}'.format(f1))
		

if __name__ == "__main__":
	data_path = "output_1/"
	start_size_num = 20

	MLP(data_path,start_size_num)
