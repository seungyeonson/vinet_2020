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