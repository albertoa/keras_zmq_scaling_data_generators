
# Copyright (c) 2018 Alberto Alonso
# MIT license. See LICENSE file for details

# Import standard python modules
import glob
import cv2
import numpy as np
import os
import json
import h5py
import math
import random

# Get the base configuration for the project
import cfg.base as cfg


def img_get_random_patch(img,w,h):
    """Get a random patch of a specific width and height from an image"""

    # Note that for this function it is the user's responsibility to ensure
    # the image size is big enough. We'll do an asertion to help but...

    # Figure out the maximum starting point within the image that 
    max_x = img.shape[1] - w
    max_y = img.shape[0] - h

    # Make sure the size is big enough
    assert max_x >= 0, 'Trying to get a patch wider that the image width'
    assert max_y >= 0, 'Trying to get a patch higher that the image height'

    # Get a random starting point
    x = random.randint(0,max_x)
    y = random.randint(0,max_y)

    # Get the patch within the image
    image_patch = img[y:y+h,x:x+w, ...]

    # All done
    return image_patch


def hdf5_get_rand_set(dbfs,cnt,img_size):
    """Get a random set of image patches (and their data) from within a set of HDF5 databases"""
    # Start with empty lists
    labels = []
    imgs = []
    manipulations = []

    # Go ahead and loop until we have the desired number
    i = 0
    while i < cnt:
        # We cycle randomly through the database files
        dbf = random.choice(dbfs)

        # Need to know how many images are in the chosen database
        db_img_count = dbf["label"].shape[0]

        # Get the index to the db
        imgidx = random.randint(0,db_img_count-1)

        # First get the label
        label = dbf["label"][imgidx].decode("ascii")

        # If the label is empty the image at this index is to be ignored
        # Somehow 938 images made it to the DB without labels. The image data
        # is either invalid or empty.
        if label == '':
            continue
        
        # We want 1/2 of the images to be manipulated
        if np.random.rand() < 0.5:
            # Choose a manipulation operation at random
            db = random.choice(cfg.manip)
            manipulated = 1.0
        else:
            # Here we select the raw image data
            db = "img"
            manipulated = 0.0

        # We want 1/2 of the images rotated
        if np.random.rand() < 0.5:
            rotate = True
        else:
            rotate = False

        # Store the label
        labels.append(label)
        # Store the manipulation flag
        manipulations.append(manipulated)

        # Store the image patch either rotated or as is
        if rotate:
            imgs.append(img_get_random_patch(dbf[db][imgidx, ...], img_size, img_size))
        else:
            imgs.append(np.rot90(img_get_random_patch(dbf[db][imgidx, ...], img_size, img_size),1,(0,1)))

        # We processed the image successfully, go to the next one
        i += 1

    # We now have a full set
    return imgs,manipulations,labels
