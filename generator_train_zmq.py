
# Copyright (c) 2018 Alberto Alonso
# MIT license. See LICENSE file for details

# Import standard python modules
import numpy as np
import argparse
import json
import os
import sys
import glob
import zmq
import random

# Image related imports
import cv2
import scipy
import scipy.misc
import skimage
import skimage.exposure

# Get the base configuration for the project
import cfg.base as cfg

# Default configuration parameters
params = {
    # Expected batch size. It is OK if it doesn't fully match
	"batch_size": 36
    # Patch width.
	,"patch_width": 224
    # Patch height.
	,"patch_height": 224
    # Patch channel depth.
	,"patch_depth": 3
}

# Override specific parameters from the configuration file
params["dataset_classes"] = cfg.dataset_classes
params["train_zmq_host"] = cfg.zmq_train_sink_host
params["train_zmq_port"] = cfg.zmq_train_sink_port

# The manipulation keys prepends the no-manipulation operation (as in '') to the
# official manipulation operations. That means that a key of 0 is no manipulation
params["manip_keys"] = [''] + cfg.manip

# List of files to use for training
fnames = glob.glob('dataset/train/*/*.*')
# Randomize the list
random.shuffle(fnames)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--zmq-dst"
				,help="ZeroMQ destination")

ap.add_argument("--batch-size", type=int
				,help="batch size")

ap.add_argument("--patch-width", type=int
				,help="patch-width")
ap.add_argument("--patch-height", type=int
				,help="patch-height")
ap.add_argument("--patch-depth", type=int
				,help="patch-depth")

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

# Come up with the ZeroMQ destination sink URL
if args["zmq_dst"] is not None:
	zmq_dst = args["zmq_dst"]
else:
	zmq_dst = "tcp://"+params["train_zmq_host"]+":"+str(params["train_zmq_port"])

# Create the ZeroMQ context
zmq_context = zmq.Context()

# Socket to send messages
zmq_send_socket = zmq_context.socket(zmq.PUSH)
# Change the high water mark to twice the batch_size (must be
# done prior to the connect)
zmq_send_socket.set_hwm(2 * params["batch_size"])
# Connect to the sink
zmq_send_socket.connect(zmq_dst)

# Allocate the message buffer
buf = np.zeros((2 + params["patch_height"] * params["patch_width"] * params["patch_depth"]),  dtype=np.uint8)

# Go into an infinite loop sending image patches
while True:
	for i,f in enumerate(fnames):
		# Get the class id
		y = params["dataset_classes"].index(f.split(os.path.sep)[-2])

		# The first byte will be the class id
		buf[0] = y

		# Figure out what manipulation we want to do (if any)
		manip_key = np.random.choice(params["manip_keys"])

		# The second byte is the manipulation operation
		buf[1] = 0

		# Load the image
		image = cv2.imread(f)

		# Ignore non color images
		if image.ndim != 3:
			continue

		# For some reason I saw a file with channel depth 4
		if image.shape[2] != 3:
			continue

		# Make sure the size is at least twice the patch size. Otherwise the bicubic resizing
		# won't be able to be done for the 0.5 scale
		if image.shape[0] < 2 * params["patch_height"]:
			continue
		if image.shape[1] < 2 * params["patch_width"]:
			continue

		# Select a patch
		x = image.shape[1] // 2 - params["patch_width"] // 2
		y = image.shape[0] // 2 - params["patch_height"] // 2
		image_patch = image[y:y+params["patch_height"],x:x+params["patch_width"],...]

		# Check for gamma correction operations
		if manip_key == 'gamma08':
			image_patch = skimage.exposure.adjust_gamma(image_patch, gamma=0.8)
		elif manip_key == 'gamma12':
			image_patch = skimage.exposure.adjust_gamma(image_patch, gamma=1.2)

		# Check for JPEG compression operations with given quality factors
		elif manip_key == 'qf90':
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
			result, encimg = cv2.imencode('.jpg', image_patch, encode_param)
			image_patch = cv2.imdecode(encimg, 1)
		elif manip_key == 'qf70':
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
			result, encimg = cv2.imencode('.jpg', image_patch, encode_param)
			image_patch = cv2.imdecode(encimg, 1)

		# Check for bicubic resize operations. These operate first on the original
		# image, then select a patch
		elif manip_key == 'bicubic05':
			tmp_image = scipy.misc.imresize(image, 0.5, interp='bicubic')
			x = tmp_image.shape[1] // 2 - params["patch_width"] // 2
			y = tmp_image.shape[0] // 2 - params["patch_height"] // 2
			image_patch = tmp_image[y:y+params["patch_height"],x:x+params["patch_width"],...]
		elif manip_key == 'bicubic08':
			tmp_image = scipy.misc.imresize(image, 0.8, interp='bicubic')
			x = tmp_image.shape[1] // 2 - params["patch_width"] // 2
			y = tmp_image.shape[0] // 2 - params["patch_height"] // 2
			image_patch = tmp_image[y:y+params["patch_height"],x:x+params["patch_width"],...]
		elif manip_key == 'bicubic15':
			tmp_image = scipy.misc.imresize(image, 1.5, interp='bicubic')
			x = tmp_image.shape[1] // 2 - params["patch_width"] // 2
			y = tmp_image.shape[0] // 2 - params["patch_height"] // 2
			image_patch = tmp_image[y:y+params["patch_height"],x:x+params["patch_width"],...]
		elif manip_key == 'bicubic20':
			tmp_image = scipy.misc.imresize(image, 2.0, interp='bicubic')
			x = tmp_image.shape[1] // 2 - params["patch_width"] // 2
			y = tmp_image.shape[0] // 2 - params["patch_height"] // 2
			image_patch = tmp_image[y:y+params["patch_height"],x:x+params["patch_width"],...]

		# Now put the image patch data into the buffer starting at the third byte
		buf[2:] = image_patch.ravel()
        
		# Time to send the data
		zmq_send_socket.send(buf,copy=True)

