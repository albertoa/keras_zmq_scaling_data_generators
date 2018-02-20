
# Copyright (c) 2018 Alberto Alonso
# MIT license. See LICENSE file for details

# Import standard python modules
import numpy as np
import argparse
import json
import os
import sys
import zmq
import h5py

# Import relevant keras modules
from keras.optimizers import Adam, SGD
from keras.models import Model
import keras.backend as K

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, \
        BatchNormalization, Activation, GlobalAveragePooling2D, SeparableConv2D, Reshape

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications import MobileNet
from keras.applications import mobilenet

# Import the appropriate project library modules
import lib.helperfunctions as hf

#############################################################################
### Configuration section
#############################################################################

# Get the base configuration for the project
import cfg.base as cfg

# Default configuration parameters
params = {
    # Path to store any output files
    "output": "models/mobilenetcustom_zmq_1"
    # HDF5 databases to use for validation
    ,"valdbpaths": [cfg.hdf5_dir+'/glebvalidationset']

    # PULL ZeroMQ port for training data
    ,"train_zmq_port": 11000

    # Training batch size
    ,"batch_size": 36
    # Number of epochs to train for
    ,"epochs": 20
    # Starting learning rate
    ,"lr": 1e-2
    # Momemtum
    ,"momentum": 0.9
    # Regularization
    ,"reg": 0.002

    # Patch width.
    ,"patch_width": 224
    # Patch height.
    ,"patch_height": 224
    # Patch channel depth.
    ,"patch_depth": 3

    # Total number of image patches we want to train on per epoch
    ,"train_numimages": 7200
    # Total number of image patches we want to validate per epoch
    ,"val_numimages": 300
}

# Use the appropriate configuration settings to override defaults
params["class_labels"] = cfg.dataset_classes
params["manip_keys"] = cfg.manip
params["train_zmq_port"] = cfg.zmq_train_sink_port
params["val_zmq_port"] = cfg.zmq_val_sink_port

#############################################################################
### Local functions
#############################################################################

def train_gen():
    """Training generator function to use by Keras. Yields batch data"""
    # Pre-allocate the arrays to hold batch data
    imgs = np.zeros((
        params["batch_size"],
        params["patch_height"],
        params["patch_width"],
        params["patch_depth"]),  dtype=np.float32)
    y = np.zeros((params["batch_size"]),  dtype=np.uint8)
    manipulations = np.zeros((params["batch_size"]),  dtype=np.float32)

    # Go into an infinite loop getting batch data
    while True:
        # Get all the images in a batch via the ZeroMQ socket
        for i in range(params["batch_size"]):
            # Get the message buffer
            msg = train_socket.recv()
            
            # Make sure the msg is of the right size, ignore it otherwise
            if len(msg) != (2 + params["patch_height"] * params["patch_width"] * params["patch_depth"]):
                print("Ignoring bad ZMQ message of size ",len(msg))
                continue

            # Get the class id
            y[i] = msg[0]
            
            # For this model we only care if there was a manipulation operation done
            # on the image, not what the actual operation was
            if msg[1] != 0:
                manipulations[i] = 1.0
            else:
                manipulations[i] = 0.0

            # Get the patch image data reshaped to the proper dimensions
            imgdata = np.frombuffer(msg[2:],np.uint8).reshape(
                params["patch_height"],params["patch_width"],params["patch_depth"])

            # Now preprocess the data as per the model needs
            imgs[i] = mobilenet.preprocess_input(imgdata.astype(np.float32))

        # We are done. We have a full batch
        #yield([[np.array(imgs),np.array(manipulations)],[np.array(y)]])
        yield([[np.array(imgs)],[np.array(y)]])

def val_gen():
    """Validation generator function to use by Keras. Yields batch data"""
    # Go into an infinite loop getting batch data
    while True:
        # Get the image data directly from the HDF5 validation databases
        imgs,manipulations,labels = hf.hdf5_get_rand_set(val_dbfs,params["batch_size"],params["patch_width"])

        # Need to go from label to class id
        y = [params["class_labels"].index(x) for x in labels]

        # Now preprocess the data as per the model needs
        for i,img in enumerate(imgs):
            imgs[i] = mobilenet.preprocess_input(img.astype(np.float32))

        # We are done. We have a full batch
        #yield([[np.array(imgs),np.array(manipulations)],[np.array(y)]])
        yield([[np.array(imgs)],[np.array(y)]])



#############################################################################
### Main section
#############################################################################
if __name__=='__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output"
                    ,help="path to output model files")
    ap.add_argument("-m", "--model", type=str
                    ,help="path to *specific* model checkpoint to load")
    ap.add_argument("-d", "--valdbpaths"
                    ,help="hdf5 validation database paths")

    ap.add_argument("--batch-size", type=int
                    ,help="batch size")
    ap.add_argument("--epochs", type=int
                    ,help="number of epochs")
    ap.add_argument("--lr", type=float
                    ,help="learning rate")
    ap.add_argument("--momentum", type=float
                    ,help="momentum to use with the optimizer")
    ap.add_argument("--reg", type=float
                    ,help="regularization")

    ap.add_argument("--patch-width", type=int
                    ,help="patch-width")
    ap.add_argument("--patch-height", type=int
                    ,help="patch-height")
    ap.add_argument("--patch-depth", type=int
                    ,help="patch-depth")

    ap.add_argument("--train_zmq_port", type=int
                    ,help="ZMQ port to listen at for training images")

    ap.add_argument("--train-numimages", type=int
                    ,help="train-numimages")
    ap.add_argument("--val-numimages", type=int
                    ,help="val-numimages")
    args = vars(ap.parse_args())

    # Handle command line overrides
    if args["output"] is not None:
        params["output"] = args["output"]
    if args["valdbpaths"] is not None:
        params["valdbpaths"] = args["valdbpaths"]

    if args["batch_size"] is not None:
        params["batch_size"] = args["batch_size"]
    if args["epochs"] is not None:
        params["epochs"] = args["epochs"]
    if args["lr"] is not None:
        params["lr"] = args["lr"]
    if args["momentum"] is not None:
        params["momentum"] = args["momentum"]
    if args["reg"] is not None:
        params["reg"] = args["reg"]

    if args["patch_width"] is not None:
        params["patch_width"] = args["patch_width"]
    if args["patch_height"] is not None:
        params["patch_height"] = args["patch_height"]
    if args["patch_depth"] is not None:
        params["patch_depth"] = args["patch_depth"]

    if args["train_zmq_port"] is not None:
        params["train_zmq_port"] = args["train_zmq_port"]

    if args["train_numimages"] is not None:
        params["train_numimages"] = args["train_numimages"]
    if args["val_numimages"] is not None:
        params["val_numimages"] = args["val_numimages"]

    # if args[""] is not None:
    #   params[""] = args[""]


    # Make sure we have the output directories or go ahead and create them
    if not os.path.exists(params["output"]):
        os.makedirs(params["output"])
    if os.path.isdir(params["output"]) == False:
        print("Couldn't create",params["output"])
    if not os.path.exists(os.path.sep.join((params["output"],"checkpoints"))):
        os.makedirs(os.path.sep.join((params["output"],"checkpoints")))
    if os.path.isdir(os.path.sep.join((params["output"],"checkpoints"))) == False:
        print("Couldn't create",os.path.sep.join((params["output"],"checkpoints")))

    # Since this project is for teaching, we'll use ZeroMQ for the
    # training data and hdf5 databases for the validation data

    # Get the ZMQ context
    context = zmq.Context()

    # Socket for getting the training data
    train_socket = context.socket(zmq.PULL)
    # Set the high water mark to 2 * batch_size
    train_socket.set_hwm(2 * params["batch_size"])
    train_socket.bind("tcp://*:"+str(params["train_zmq_port"]))

    # Open the HDF5 validation databases
    val_dbfs = []
    for fname in params["valdbpaths"]:
        val_dbfs.append(h5py.File(fname, mode='r'))

    # Save the active parameters of the training session
    with open(os.path.sep.join((params["output"],"trainingparams")), 'w') as fp:
        json.dump(params, fp)
        fp.close()

    # construct the set of callbacks
    name_format = "epoch-{epoch:03d}-val_acc-{val_acc:.4f}.hdf5"
    callbacks = [
        ModelCheckpoint(os.path.sep.join((params["output"],"checkpoints",name_format))
                        , monitor='val_acc', verbose=0
                        , save_best_only=True
                        , save_weights_only=False, mode='auto', period=1)
        , ReduceLROnPlateau(monitor='val_acc'
                            , factor=0.5, patience=10, min_lr=5e-6, epsilon = 0.00001
                            , verbose=1, mode='max')

    ]

    # Setup the model input for our image patches
    model_input = Input(shape=(params["patch_height"]
                               ,params["patch_width"]
                               ,params["patch_depth"]))

    # Setup the base model for Mobilenet, we'll have to do our own FC layers
    # to match our number of classes
    basemodel = MobileNet(include_top=False, weights='imagenet'
                          ,input_shape=(params["patch_height"]
                                        ,params["patch_width"]
                                        ,params["patch_depth"]))

    # Start building the custom top
    x = basemodel(model_input)
    x = Reshape((-1,))(x)
    # First fully connected layer and a dropout to reduce overfitting
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3, name='dropout_fc1')(x)
    # Second fully connected layer, this time we'll skip the dropout
    x = Dense(128, activation='relu', name='fc2')(x)

    # Add the prediction layer via softmax
    prediction = Dense(len(params["class_labels"])
                       ,activation ="softmax"
                       ,name="predictions")(x)

    # We are done with the model architecture
    model = Model(inputs=(model_input), outputs=prediction)

    # Always a good idea to show the model we are training
    model.summary()

    # Add the optimizer and compile the model
    opt = Adam(lr=params["lr"])
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the network
    model.fit_generator(
        train_gen()
        , steps_per_epoch=params["train_numimages"] // params["batch_size"]
        , validation_data=val_gen()
        , validation_steps=params["val_numimages"] // params["batch_size"]
        , epochs=params["epochs"]
        , max_queue_size=params["batch_size"] * 2
        , callbacks=callbacks
        , verbose=1
    )

    # All done, save the full trained model
    model.save(os.path.sep.join((params["output"],"trainedmodel"))) 
