#from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import tree
import pydot
import csv
import six
import math
from decimal import Decimal
import os
import sys
import ctypes as C
from scipy import stats


np.set_printoptions(threshold=sys.maxsize)

import XlRFInference
import pickle




enable_training_CPU = True
enable_inference_CPU = True
enable_inference_FPGA = True


bitstreamName = "/app/Xl_rf_inference.xclbin"
modelDirOffset = ""

#set the number of trees
classifierOrRegressor = 0 # Classifier=0, Regressor=1
number_of_trees = 10000
max_depth = 8
numSamples = 10

dataset = '/app/temps.csv'
key = 'actual'

#dataset = './dataset/boolean_data_airline.csv'
#key = 'isDelayed'





if (1 < len(sys.argv)):
    bitstreamName = sys.argv[1]

if (2 < len(sys.argv)):
    modelDirOffset = sys.argv[2]

if (3 < len(sys.argv)):
    number_of_trees = int(sys.argv[3])

if (4 < len(sys.argv)):
    classifierOrRegressor = int(sys.argv[4])

if (5 < len(sys.argv)):
    numSamples = int(sys.argv[5])



#prepare filename according to parameters
predictionType = "classification"
if (1 == classifierOrRegressor):
    predictionType = "regression"



filename = modelDirOffset + "model_" + str(predictionType) + "_" + str(number_of_trees) + "_" + str(max_depth) + ".pkl"

# Read in data and display first 5 rows
features = pd.read_csv(dataset)
print(features.head(5))
print('The shape of our features is:', features.shape)

print('apply one-hot encoding for categorical variables ...')
# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
print(features.head(5))
print('The shape of our features is:', features.shape)

# Labels are the values we want to predict
labels = np.array(features[key])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop(key, axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Split the data into training and testing sets, and artificially expand it
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 12345)
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = np.concatenate([test_features,test_features, test_features, test_features])
test_labels = np.concatenate([test_labels,test_labels,test_labels,test_labels])
test_features = test_features[0:numSamples]
test_labels = test_labels[0:numSamples]

print(" NUM trees = ", number_of_trees)
print(" NUM samples =", test_labels.shape[0])
print(" prediction type: ", predictionType)



if classifierOrRegressor == 0:
    clf = RandomForestClassifier(bootstrap=None, class_weight=None, criterion='gini',
                    max_depth=max_depth, max_features=features.shape[1], max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, n_estimators=number_of_trees, n_jobs=32,
                    oob_score=False, random_state=1234, verbose=1, warm_start=False)

if classifierOrRegressor == 1:
    clf = RandomForestRegressor(max_depth=max_depth, max_features=features.shape[1], max_leaf_nodes=None,
                            n_estimators=number_of_trees, n_jobs=32, random_state=1234)

if (enable_training_CPU):
    start_time = time.perf_counter()
    clf.fit(train_features, train_labels)
    print("training time is", time.perf_counter() - start_time, "seconds")

    #save model to file
    pickle.dump(clf, open(filename, 'wb'))

if ((not enable_inference_CPU) and (not enable_inference_FPGA)):
    exit()

clf = pickle.load(open(filename, 'rb'))

tree_in_forest = clf.estimators_[0]
value = tree_in_forest.tree_.value
num_classes=value[0].shape[1]

# time measurements
sw_time = None
HW_setup_time = None
HW_infer_time = None

SW_predictions = None
sw_start_time = None
sw_end_time = None

if (enable_inference_CPU):
    if (num_classes == 1):
        sw_start_time = time.perf_counter()
        SW_predictions = clf.predict(test_features)
        sw_end_time = time.perf_counter()
    else:
        sw_start_time = time.perf_counter()
        SW_predictions = clf.predict_proba(test_features)
        sw_end_time = time.perf_counter()

    sw_time = sw_end_time - sw_start_time

    print('Finished CPU inference/prediction')


###############################################################
######## HW part ##############################################
if (enable_inference_FPGA):

    hw_formatting_time_start = time.perf_counter()
    xlrfsetup = XlRFInference.XlRFSetup()
    xlrfsetup.setTrees(clf)


    #retrieve parameters from model
    params = xlrfsetup.getModelParameters()


    hw_formatting_time_end = time.perf_counter()

    HW_formatting_time = hw_formatting_time_end - hw_formatting_time_start


    setup_starttime = time.perf_counter()

    xlrf = XlRFInference.XlRFInference(bitstreamName)

    #re-feed trees to inference engine and transfer to board

    xlrf.setModelParameters(params)

    setup_endtime = time.perf_counter()
    HW_setup_time = setup_endtime - setup_starttime


    nLoops = 100
    if (numSamples < 1000):
        nLoops = 1000

    hw_infer_start_time = time.perf_counter()
    for i in range(nLoops):
        HW_predictions = xlrf.predict(test_features, len(clf.estimators_))

    hw_infer_end_time = time.perf_counter()
    HW_infer_time = hw_infer_end_time - hw_infer_start_time
    HW_infer_time /= nLoops

    print('Finished FPGA inference/prediction')


# Calculate the absolute errors
if (num_classes != 1):
    if (enable_inference_FPGA and enable_inference_CPU):
        errors = abs(SW_predictions - HW_predictions)
    if (enable_inference_CPU):
        SW_predictions =SW_predictions.argmax(1)
        SW_predictions = clf.classes_.take(SW_predictions, axis=0)
    if (enable_inference_FPGA):
        HW_predictions =HW_predictions.argmax(1)
        HW_predictions = clf.classes_.take(HW_predictions, axis=0)


if (enable_inference_CPU):
    SW_errors = abs(SW_predictions - test_labels)
    print('SW vs Gold: Mean Absolute Error:', round(np.mean(SW_errors), 2), 'degrees.')
if (enable_inference_CPU):
    HW_errors = abs(HW_predictions - test_labels)
    print('HW vs Gold: Mean Absolute Error:', round(np.mean(HW_errors), 2), 'degrees.')

if (num_classes != 1):
    if (enable_inference_FPGA and enable_inference_CPU):
        print('HW vs SW: Mean Absolute Error:', (round(np.mean(errors), 4))*100, '%')



print("")


print('LOGINFO bitstream =              ', bitstreamName)
print('LOGINFO NUM trees =              ', number_of_trees)
print('LOGINFO NUM samples =            ', test_labels.shape[0])
print('LOGINFO prediction type:         ', predictionType)
if (enable_inference_CPU):
    print('LOGINFO SW predict time:         ', sw_time, 's')
if (enable_inference_CPU):
    print('LOGINFO HW_formatting_time:      ', HW_formatting_time, 's')
    print('LOGINFO HW_setup time:           ', HW_setup_time, 's')
    print('LOGINFO HW_infer time (', nLoops, 'x avg):', HW_infer_time, 's')
if (enable_inference_FPGA and enable_inference_CPU):
    print('LOGINFO HW vs.SW speedup:        ', sw_time/HW_infer_time, 'X')
print('LOGINFO ')
