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

from model import VINet
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
from Dataloader import Dataloader
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
	pass
	# Model definition without batchnorm
	VINet = VINet(arg.imageWidth, arg.imageHeight, activation = arg.activation, \
	 	numLSTMCells = arg.numLSTMCells, hidden_units_LSTM = [1024, 1024])
elif arg.modelType == 'vinet_batchnorm':
	pass
	# Model definition with batchnorm
	# deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization, \
	# 	batchnorm = True, flownet_weights_path = cmd.loadModel)
elif arg.modelType == 'our_model':
	pass

# Load a pretrained DeepVO model
if arg.modelType == 'vinet':
	# deepVO = torch.load(cmd.loadModel)
	pass
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
rotLosses_train = []
transLosses_train = []
totalLosses_train = []
rotLosses_val = []
transLosses_val = []
totalLosses_val = []

f2fLosses_train = []
f2fLosses_val = []
bestValLoss = np.inf


# Create datasets for the current epoch
# train_seq = [0, 1, 2, 8, 9]
# train_startFrames = [0, 0, 0, 0, 0]
# train_endFrames = [4540, 1100, 4660, 4070, 1590]
# val_seq = [3, 4, 5, 6, 7, 10]
# val_startFrames = [0, 0, 0, 0, 0, 0]
# val_endFrames = [800, 270, 2760, 1100, 1100, 1200]

#for test
train_seq = [1]
train_startFrames = [0]
train_endFrames = [4161]
val_seq = [0]
val_startFrames = [0]
val_endFrames = [2551]


for epoch in range(arg.nepochs):
	
	print('===============> Starting epoch: '+str(epoch+1) + '/'+str(arg.nepochs))

	train_seq_cur_epoch = []
	train_startFrames_cur_epoch = []
	train_endFrames_cur_epoch = []
	# Take each sequence and split it into chunks
	for s in range(len(train_seq)):
		train_seq_cur_epoch.append(train_seq[s])
		train_startFrames_cur_epoch.append(train_startFrames[s])
		train_endFrames_cur_epoch.append(train_endFrames[s])

	#train split by sequence
	permutation = np.random.permutation(len(train_seq_cur_epoch))
	train_seq_cur_epoch = [train_seq_cur_epoch[p] for p in permutation]
	train_startFrames_cur_epoch = [train_startFrames_cur_epoch[p] for p in permutation]
	train_endFrames_cur_epoch = [train_endFrames_cur_epoch[p] for p in permutation]

	train_data = Dataloader(arg.datadir, train_seq_cur_epoch, train_startFrames_cur_epoch, \
		train_endFrames_cur_epoch, width = arg.imageWidth, height = arg.imageHeight)
	val_data = Dataloader(arg.datadir, val_seq, val_startFrames, val_endFrames, \
		width = arg.imageWidth, height = arg.imageHeight)

	# Initialize a trainer (Note that any accumulated gradients on the model are flushed
	# upon creation of this Trainer object)
	trainer = Trainer(arg, epoch, deepVO, train_data, val_data, criterion, optimizer, \
					  scheduler=None)

	# Training loop
	print('===> Training: ' + str(epoch + 1) + '/' + str(arg.nepochs))
	startTime = time.time()
	rotLosses_train_cur, transLosses_train_cur, totalLosses_train_cur = trainer.train()
	print('Train time: ', time.time() - startTime)

	rotLosses_train += rotLosses_train_cur
	transLosses_train += transLosses_train_cur
	totalLosses_train += totalLosses_train_cur