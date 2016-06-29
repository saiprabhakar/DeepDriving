#!/usr/bin/env sh

TOOLS=../build/tools

# 2 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_2lane.bin pre_trained/driving_run_1F.prototxt pre_trained/driving_train_1F_iter_140000.caffemodel GPU

# 3 lane
GLOG_logtostderr=1 $TOOLS/torcs_run_3lane.bin pre_trained/driving_run_1F.prototxt pre_trained/driving_train_1F_iter_140000.caffemodel GPU

# 1 lane
#GLOG_logtostderr=1 $TOOLS/torcs_run_1lane.bin pre_trained/driving_run_1F.prototxt pre_trained/driving_train_1F_iter_140000.caffemodel GPU

