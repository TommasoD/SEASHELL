
# for running the following script : c--, python3 Colab_Ml-image..........


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
#from imutils import paths
import numpy as np
import pickle
import random
import os
import pandas as pd



def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
	# open the input file for reading
	f = open(inputPath, "r")
	# loop indefinitely
	while True:
		# initialize our batch of data and labels
		data = []
		labels = []
		# keep looping until we reach our batch size
		while len(data) < bs:
			# attempt to read the next row of the CSV file
			row = f.readline()
			# check to see if the row is empty, indicating we have
			# reached the end of the file
			if row == "":
				# reset the file pointer to the beginning of the file
				# and re-read the row
				f.seek(0)
				row = f.readline()
				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break
			# extract the class label and features from the row
			row = row.strip().split(",")
			label = row[0]
			label = to_categorical(label, num_classes=numClasses)
			features = np.array(row[1:], dtype="float")
			# update the data and label lists
			data.append(features)
			labels.append(label)
			#list = []
			#list.append(np.array(data))
			#list.append(np.array(labels))
		# yield the batch to the calling function
		yield (np.array(data),np.array(labels))
		#return list

#TODO: COUNT THE TOTAL NUMBER OF IMAGES per vedere se le ha presee tutte (forse solo 32)


# load the label encoder from disk
#le = pickle.loads(open("/opt/spark-data/output/le.cpickle", "rb").read())
le = pickle.loads(open("/home/lorenzoamata/Scaricati/docker-hadoop-master/data/output/le.cpickle", "rb").read())
# derive the paths to the training, validation, and testing CSV files

#-------------------------------FIXA CON OPT SPARK DATA PATH CORRETTO
trainPath = "/home/lorenzoamata/Scaricati/docker-hadoop-master/data/train.csv"
valPath = "/home/lorenzoamata/Scaricati/docker-hadoop-master/data/val.csv"
testPath ="/home/lorenzoamata/Scaricati/docker-hadoop-master/data/test.csv"
# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])
# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)


trainGen= csv_feature_generator(trainPath, 32,
	2, mode="train")
valGen = csv_feature_generator(valPath, 32,
	2, mode="eval")
testGen = csv_feature_generator(testPath, 32,
	2, mode="eval")

model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
# compile the model
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training simple network...")
H = model.fit(
	x=trainGen,
	steps_per_epoch=totalTrain // 32,
	validation_data=valGen,
	validation_steps=totalVal // 32,
	epochs=25)
# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict(x=testGen,
	steps=(totalTest //32) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
	target_names=le.classes_))
