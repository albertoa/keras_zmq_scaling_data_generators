
# Based on which machine we are running from we need to
# update where our data is as well as where to connect to
# when using a distributed framework
import socket
hn = socket.gethostname()
if hn == "w075":
	# Set the directory where all the dataset links are
	link_dir = "nvmelinks/"
	# This is the GPU machine. Use localhost for connections
	zmq_train_sink_host = "localhost"
	zmq_val_sink_host = "localhost"
else:
	# This is a container or a remote machine. All dataset links
	# are in dockerlinks
	link_dir = "dockerlinks/"
	# Set the destination GPU machine
	zmq_train_sink_host = "w075"
	zmq_val_sink_host = "w075"

# Set the ZeroMQ ports for training and validation sinks
zmq_train_sink_port = 12001
zmq_val_sink_port = 12002

# Configure the dataset directories
hdf5_dir       = link_dir+'hdf5'
train_dir       = link_dir+'train'
extra_train_dir = link_dir+'flickr_images'
extra_val_dir   = link_dir+'val_images'
test_dir        = link_dir+'test'

# Where to store model data
model_dir       = 'models'
# Where to store prediction data
pred_dir       = 'preds'
# Where to store submission data
sub_dir       = 'submissions'

# This must match the order used by the models. Class id will be the
# index of the label within this list
dataset_classes = [
    'HTC-1-M7',
    'iPhone-6',     
    'Motorola-Droid-Maxx',
    'Motorola-X',
    'Samsung-Galaxy-S4',
    'iPhone-4s',
    'LG-Nexus-5x', 
    'Motorola-Nexus-6',
    'Samsung-Galaxy-Note3',
    'Sony-NEX-7'
]

# Manipulation keys (note that they must match the databases within the hdf5 files as
# well as manip keys from any custom generators)
manip = ['gamma08', 'gamma12', 'qf90', 'qf70', 'bicubic05', 'bicubic08', 'bicubic15', 'bicubic20']
