"""
Main script: Train and Test VINet on the euroc, uzh, etc...
"""

# The following two lines are needed because, conda on Mila SLURM sets
# 'Qt5Agg' as the default version for matplotlib.use(). The interpreter
# throws a warning that says matplotlib.use('Agg') needs to be called
# before importing pyplot. If the warning is ignored, this results in
# an error and the code crashes while storing plots (after validation).
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Other project files with definitions
import args

#Parse arguments
arg = args.arguments

# Debug parameters. This is to run in 'debug' mode, which runs a very quick pass
# through major segments of code, to ensure nothing awful happens when we deploy
# on GPU clusters for instance, as a batch script. It is sometimes very annoying
# when code crashes after a few epochs of training, while attempting to write a
# checkpoint to a directory that does not exist.
if arg.debug is True:
	arg.debugIters = 3
	arg.nepochs = 2

# Set default tensor type to cuda.FloatTensor, for GPU execution
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Setting directory and Create result directory structure, to store results
arg.basedir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(arg.basedir, arg.resultdir, arg.dataset)):
	os.makedirs(os.path.join(arg.basedir, arg.resultdir, arg.dataset))

#if you want to try multiple, change expID
arg.expDir = os.path.join(arg.basedir, arg.resultdir, arg.dataset, arg.expID)
if not os.path.exists(arg.expDir):
	os.makedirs(arg.expDir)
	print('Created dir: ', arg.expDir)
if not os.path.exists(os.path.join(arg.expDir, 'models')):
	os.makedirs(os.path.join(arg.expDir, 'models'))
	print('Created dir: ', os.path.join(arg.expDir, 'models'))
if not os.path.exists(os.path.join(arg.expDir, 'plots', 'traj')):
	os.makedirs(os.path.join(arg.expDir, 'plots', 'traj'))
	print('Created dir: ', os.path.join(arg.expDir, 'plots', 'traj'))
if not os.path.exists(os.path.join(arg.expDir, 'plots', 'loss')):
	os.makedirs(os.path.join(arg.expDir, 'plots', 'loss'))
	print('Created dir: ', os.path.join(arg.expDir, 'plots', 'loss'))
if arg.dataset == 'uzh':
	number = 27+1
elif arg.dataset =='euroc':
	number = 10+1
else:
	#If you want to add dataset, you give number with dataset sequence length
	raise Exception
for seq in range(number):
	if not os.path.exists(os.path.join(arg.expDir, 'plots', 'traj', str(seq).zfill(2))):
		os.makedirs(os.path.join(arg.expDir, 'plots', 'traj', str(seq).zfill(2)))
		print('Created dir: ', os.path.join(arg.expDir, 'plots', 'traj', str(seq).zfill(2)))

# Save all the command line arguements in a text file in the experiment directory.
argFile = open(os.path.join(arg.expDir, 'args.txt'), 'w')
for i in vars(arg):
	argFile.write(i + ' ' + str(getattr(arg, i)) + '\n')
argFile.close()

# TensorboardX visualization support
if arg.tensorboardX is True:
	from tensorboardX import SummaryWriter
	writer = SummaryWriter(log_dir = arg.expDir)

########################################################################
### Model Definition + Weight init + FlowNet weight loading ###
########################################################################

# Get the definition of the model
if arg.modelType == 'vinet' or arg.modelType is None:
	# Model definition without batchnorm
	# deepVO = DeepVO(cmd.imageWidth, cmd.imageHeight, activation = cmd.activation, parameterization = cmd.outputParameterization, \
	# 	numLSTMCells = cmd.numLSTMCells, hidden_units_LSTM = [1024, 1024])
elif arg.modelType == 'vinet_batchnorm':
	# Model definition with batchnorm
	# deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization, \
	# 	batchnorm = True, flownet_weights_path = cmd.loadModel)
elif arg.modelType == 'our_model':
	pass

# Load a pretrained DeepVO model
if arg.modelType == 'vinet':
	# deepVO = torch.load(cmd.loadModel)
else:
	# Initialize weights for fully connected layers and for LSTMCells
	# deepVO.init_weights()
	# CUDAfy
	# deepVO.cuda()
	pass
print('Loaded! Good to launch!')

########################################################################
### Criterion, optimizer, and scheduler ###
########################################################################



########################################################################
###  Main loop ###
########################################################################

