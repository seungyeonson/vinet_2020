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
from tensorboardX import SummaryWriter
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
from Trainer import Trainer
from data_info import DataInfo

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

	writer = SummaryWriter(log_dir = arg.expDir)

########################################################################
### Model Definition + Weight init + FlowNet weight loading ###
########################################################################

# Get the definition of the model
if arg.modelType == 'vinet' or arg.modelType is None:
	pass
	# Model definition without batchnorm
	VINet = VINet(arg.imageWidth, arg.imageHeight, activation = arg.activation, \
	 	numLSTMCells = arg.numLSTMCells, hidden_units_imu=[6,6], hidden_units_LSTM = [1024, 1024], batchnorm=False)
elif arg.modelType == 'vinet_batchnorm':
	pass
	# Model definition with batchnorm
	# deepVO = DeepVO(activation = cmd.activation, parameterization = cmd.outputParameterization, \
	# 	batchnorm = True, flownet_weights_path = cmd.loadModel)
elif arg.modelType == 'our_model':
	pass

# Load a pretrained DeepVO model
if arg.loadModel == 'vinet':
	#TODO: MODEL LOAD WEIGHT
	# deepVO = torch.load(cmd.loadModel)
	pass
else:
	# Initialize weights for fully connected layers and for LSTMCells
	VINet.init_weights()
	# CUDAfy
	VINet.cuda()


# TODO : Check1
if arg.loadModelCNN is not None :
	pretrained_w = torch.load(arg.loadModelCNN)
	pretrained_dict = pretrained_w['state_dict']
	model_dict = VINet.state_dict()

	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	VINet.load_state_dict(model_dict)

print('Loaded! Good to launch!')

########################################################################
### Criterion, optimizer, and scheduler ###
########################################################################
criterion = nn.MSELoss(reduction = 'sum')

if arg.optMethod == 'adam':
	optimizer = optim.Adam(VINet.parameters(), lr = arg.lr, betas = (arg.beta1, arg.beta2), weight_decay = arg.weightDecay, amsgrad = False)
elif arg.optMethod == 'sgd':
	optimizer = optim.SGD(VINet.parameters(), lr = arg.lr, momentum = arg.momentum, weight_decay = arg.weightDecay, nesterov = False)
else:
	optimizer = optim.Adagrad(VINet.parameters(), lr = arg.lr, lr_decay = arg.lrDecay , weight_decay = arg.weightDecay)

#Initialized schduler,
if arg.lrScheduler is not None:
	if arg.lrScheduler == 'cosine':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arg.nepochs)
	elif arg.lrScheduler == 'plateau':
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

########################################################################
###  Main loop ###
########################################################################
r6Losses_train = []
poseLosses_train = []
totalLosses_train = []
r6Losses_val = []
poseLosses_val = []
totalLosses_val = []
bestValLoss = np.inf


# Create datasets for the current epoch
info_dict = DataInfo()
train_seq = [0]#, 1, 2]#, 3, 4, 8, 9]
train_startFrames = info_dict.get_startFrames(train_seq)
train_endFrames = info_dict.get_endFrames(train_seq)
val_seq = [5]#, 6]#, 7, 10]
val_startFrames = info_dict.get_startFrames(val_seq)
val_endFrames = info_dict.get_endFrames(val_seq)

# #for test
# train_seq = [1]
# train_startFrames = [0]
# train_endFrames = [4161]
# val_seq = [0]
# val_startFrames = [0]
# val_endFrames = [2551]


for epoch in range(arg.nepochs):
	#TODO : 에폭이 지나가면 왜 가중치는 그대로인지... 원인을 파악하고 해결해야한다.
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
	print(train_seq_cur_epoch)
	train_startFrames_cur_epoch = [train_startFrames_cur_epoch[p] for p in permutation]
	train_endFrames_cur_epoch = [train_endFrames_cur_epoch[p] for p in permutation]

	train_data = Dataloader(arg.datadir, train_seq_cur_epoch, train_startFrames_cur_epoch, \
		train_endFrames_cur_epoch, width = arg.imageWidth, height = arg.imageHeight)
	val_data = Dataloader(arg.datadir, val_seq, val_startFrames, val_endFrames, \
		width = arg.imageWidth, height = arg.imageHeight)

	# Initialize a trainer (Note that any accumulated gradients on the model are flushed
	# upon creation of this Trainer object)
	trainer = Trainer(arg, epoch, VINet, train_data, val_data, criterion, optimizer, \
					  scheduler=None)


	# # Training loop
	print('===> Training: ' + str(epoch + 1) + '/' + str(arg.nepochs))
	startTime = time.time()
	r6Losses_train_cur, poseLosses_train_cur, totalLosses_train_cur = trainer.train()
	print('r6Losees_train_cur :', r6Losses_train_cur)
	print('poseLosses_train_cur :', poseLosses_train_cur)
	print('totalLosses_train_cur :', totalLosses_train_cur)
	print('Train time: ', time.time() - startTime)

	r6Losses_train += r6Losses_train_cur
	poseLosses_train += poseLosses_train_cur
	totalLosses_train += totalLosses_train_cur

	if arg.lrScheduler is not None:
		scheduler.step()
	# Snapshot
	if arg.snapshotStrategy == 'default' or 'best':
		if epoch % arg.snapshot == 0 or epoch == arg.nepochs - 1:
			print('Saving model after epoch', epoch, '...')
			torch.save(VINet, os.path.join(arg.expDir, 'models', 'model' + str(epoch).zfill(3) + '.pt'))
	elif arg.snapshotStrategy == 'recent':
		# Save the most recent model
		print('Saving model after epoch', epoch, '...')
		torch.save(VINet, os.path.join(arg.expDir, 'models', 'recent.pt'))
	elif arg.snapshotStrategy == 'best' or 'none':
		# If we only want to save the best model, defer the decision
		pass

	# Validation loop
	print('===> Validation: '  + str(epoch+1) + '/' + str(arg.nepochs))
	startTime = time.time()
	r6Losses_val_cur, poseLosses_val_cur, totalLosses_val_cur = trainer.validate()
	print('Val time: ', time.time() - startTime)

	r6Losses_val += r6Losses_val_cur
	poseLosses_val += poseLosses_val_cur
	totalLosses_val += totalLosses_val_cur
	# Snapshot (if using 'best' strategy)
	if arg.snapshotStrategy == 'best':
		if np.mean(totalLosses_val_cur) <= bestValLoss:
			bestValLoss = np.mean(totalLosses_val_cur)
			print('Saving recent best model after epoch', epoch, '...')
			torch.save(VINet, os.path.join(arg.expDir, 'models', 'best' + '.pt'))
	if arg.tensorboardX is True:
		writer.add_scalar('loss/train/r6_loss_train', np.mean(r6Losses_train), epoch)
		writer.add_scalar('loss/train/pose_loss_train', np.mean(poseLosses_train), epoch)
		writer.add_scalar('loss/train/total_loss_train', np.mean(totalLosses_train), epoch)
		writer.add_scalar('loss/train/r6_loss_val', np.mean(r6Losses_val), epoch)
		writer.add_scalar('loss/train/pose_loss_val', np.mean(poseLosses_val), epoch)
		writer.add_scalar('loss/train/total_loss_val', np.mean(totalLosses_val), epoch)
		writer.flush()

	# Save training curves
	fig, ax = plt.subplots(1)
	ax.plot(range(len(r6Losses_train)), r6Losses_train, 'r', label = 'rot_train')
	ax.plot(range(len(poseLosses_train)), poseLosses_train, 'g', label = 'trans_train')
	ax.plot(range(len(totalLosses_train)), totalLosses_train, 'b', label = 'total_train')
	ax.legend()
	plt.ylabel('Loss')
	plt.xlabel('Batch #')
	fig.savefig(os.path.join(arg.expDir,'plots','loss','loss_train_' + str(epoch).zfill(3)))

	fig, ax = plt.subplots(1)
	ax.plot(range(len(r6Losses_val)), r6Losses_val, 'r', label = 'rot_train')
	ax.plot(range(len(poseLosses_val)), poseLosses_val, 'g', label = 'trans_val')
	ax.plot(range(len(totalLosses_val)), totalLosses_val, 'b', label = 'total_val')
	ax.legend()
	plt.ylabel('Loss')
	plt.xlabel('Batch #')
	fig.savefig(os.path.join(arg.expDir,'plots','loss', 'loss_val_' + str(epoch).zfill(3)))

# Plot trajectories (validation sequences)
	i = 0
	for s in val_seq:
		seqLen = val_endFrames[i] - val_startFrames[i]
		trajFile = os.path.join(arg.expDir, 'plots', 'traj', str(s).zfill(2), \
			'traj_' + str(epoch).zfill(3) + '.txt')
		if os.path.exists(trajFile):
			traj = np.loadtxt(trajFile)
			traj = traj[:,3:]
			if arg.outputFrame == 'local':
				# plotSequenceRelative(arg.expDir, s, seqLen, traj, arg.datadir, arg, epoch)
				pass
			elif arg.outputFrame == 'global':
				# plotSequenceAbsolute(arg.expDir, s, seqLen, traj, arg.datadir, arg, epoch)
				pass
		i += 1


print('Done !!')