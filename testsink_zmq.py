
# Copyright (c) 2018 Alberto Alonso
# MIT license. See LICENSE file for details

# Import standard python modules
import numpy as np
import argparse
import json
import os
import sys
import zmq
import cv2
import time

# Get the base configuration for the project
import cfg.base as cfg

# Default parameters
params = {
    # How many batches to get
	"batch_count": 3
    # Image patches per batch (for the testsink it should be be a
    # multiple of 10, since we are stacking the images based on 10)
	,"batch_size": 50

    # Default training data port
	,"train_zmq_port": 11000

    # Patch width. Must match the generator
	,"patch_width": 224
    # Patch height. Must match the generator
	,"patch_height": 224
    # Patch channel depth. Must match the generator
	,"patch_depth": 3
}

# Use the appropriate configuration settings to override defaults
params["class_labels"] = cfg.dataset_classes
params["train_zmq_port"] = cfg.zmq_train_sink_port

# The manipulation keys prepends the no-manipulation operation (as in '') to the
# official manipulation operations. That means that a key of 0 is no manipulation
params["manip_keys"] = [''] + cfg.manip

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--batch-size", type=int
				,help="batch size")

ap.add_argument("--patch-width", type=int
				,help="patch-width")
ap.add_argument("--patch-height", type=int
				,help="patch-height")
ap.add_argument("--patch-depth", type=int
				,help="patch-depth")

ap.add_argument("--train_zmq_port", type=int
				,help="ZMQ port to listen at for training images")
args = vars(ap.parse_args())

# Handle command line overrides
if args["batch_size"] is not None:
	params["batch_size"] = args["batch_size"]

if args["patch_width"] is not None:
	params["patch_width"] = args["patch_width"]
if args["patch_height"] is not None:
	params["patch_height"] = args["patch_height"]
if args["patch_depth"] is not None:
	params["patch_depth"] = args["patch_depth"]

if args["train_zmq_port"] is not None:
	params["train_zmq_port"] = args["train_zmq_port"]

# For the fake sink make the batch_size a multiple of 10 so
# that we can display the patches easily stacked
params["batch_size"] = params["batch_size"] // 10 * 10

# Get a ZMQ context
context = zmq.Context()

# Socket for getting the training data
train_socket = context.socket(zmq.PULL)
# Set the high water mark to 2 * batch_size
train_socket.set_hwm(2 * params["batch_size"])
train_socket.bind("tcp://*:"+str(params["train_zmq_port"]))

# Pre-allocate the batch data arrays
imgs = np.zeros((params["batch_size"], params["patch_height"], params["patch_width"], params["patch_depth"]),  dtype=np.uint8)
y = np.zeros((params["batch_size"]),  dtype=np.uint8)
ops = np.zeros((params["batch_size"]),  dtype=np.uint8)

# Go ahead and read the batches
for batchidx in range(params["batch_count"]):
	# Start our timing
	tstart = time.time()

    # Get all the images in a batch
	for i in range(params["batch_size"]):
        # Get the message buffer
		msg = train_socket.recv()

		# Make sure the msg is of the right size, ignore it otherwise
		if len(msg) != (2 + params["patch_height"] * params["patch_width"] * params["patch_depth"]):
			print("Ignoring bad ZMQ message of size ",len(msg))
			continue

		# Get the class id
		y[i] = msg[0]
		# Get the operation code
		ops[i] = msg[1]

		# Get the patch image data reshaped to the proper dimensions
		imgs[i] = np.frombuffer(msg[2:],np.uint8).reshape(
			params["patch_height"],params["patch_width"],params["patch_depth"])

	# Figure out how long it took to get the batch data
	tend = time.time()
	print("Get batch took: %d msec" % ((tend-tstart)*1000))

    # Create a contact sheet of all the images and display it
	imgstack = []
	for i in range(params["batch_size"] // 10):
		imgstack.append(np.hstack(imgs[i*10:i*10+10]))
	img = np.vstack(imgstack)
	cv2.imshow("imgbatch",img)
	k = cv2.waitKey(0)
    # Allow early termination
	if k & 0xFF == ord('q'):
		break
	elif k & 0xFF == ord('Q'):
		break
	elif k & 0xFF == 27: # Also check for escape key
		break
