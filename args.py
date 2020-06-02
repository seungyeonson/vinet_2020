# arguments

import argparse
parser = argparse.ArgumentParser()

##paths option
parser.add_argument('-datadir', help = 'Absolute path to the directory that holds the dataset', \
	type = str, default = '/home/ssy/workspace/dataset')
parser.add_argument('-resultdir', \
	help = '(Relative path to) directory in which to store logs, models, plots, etc.', \
	type = str, default = 'result')

##Model option
parser.add_argument('-loadModel', help='load pretrained weights, if you want add pretrained weights, give path, else None',default=None)
parser.add_argument('-modelType', help='Type of the model to be loaded:1. vinet_batchnorm 2. vinet 3. our_model', type = str.lower, \
                    choices = ['vinet_batchnorm', 'vinet', 'our_model'],default = 'vinet')
parser.add_argument('-activation', help='Actication function to be used', type = str.lower, choices=['relu','selu'], default= 'relu')
parser.add_argument('-imageWidth', help = 'Width of the input image', type = int, default = 640)
parser.add_argument('-imageHeight', help = 'Height of the input image', type = int, default = 192)

##Dataset option
parser.add_argument('-dataset', help='dataset to be used for training the network, 1. uzh, 2. euroc, 3. ',choices=['uzh','euroc',''], default = 'uzh')

##Hyperparameter
parser.add_argument('-lr', help = 'Learning rate', type = float, default = 1e-4)
parser.add_argument('-momentum', help = 'Momentum', type = float, default = 0.009)
parser.add_argument('-weightDecay', help = 'Weight decay', type = float, default = 0.)
parser.add_argument('-lrDecay', help = 'Learning rate decay factor', type = float, default = 0.)
parser.add_argument('-iterations', help = 'Number of iterations after loss is to be computed', \
	type = int, default = 100)
parser.add_argument('-beta1', help = 'beta1 for ADAM optimizer', type = float, default = 0.9)
parser.add_argument('-beta2', help = 'beta2 for ADAM optimizer', type = float, default = 0.999)
parser.add_argument('-gradClip', help = 'Max allowed magnitude for the gradient norm, \
	if gradient clipping is to be performed. (Recommended: 1.0)', type = float)
parser.add_argument('-crit', help = 'Error criterion', default = 'MSE')
parser.add_argument('-optMethod', help = 'Optimization method : adam | sgd | adagrad ', \
	type = str.lower, choices = ['adam', 'sgd', 'adagrad'], default = 'adam')
parser.add_argument('-lrScheduler', help = 'Learning rate scheduler', type = str.lower, \
	choices = ['cosine', 'plateau'])

parser.add_argument('-nepochs', help = 'Number of epochs', type = int, default = 20)
parser.add_argument('-trainBatch', help = 'train batch size', type = int, default = 40)
parser.add_argument('-validBatch', help = 'valid batch size', type = int, default = 1)

parser.add_argument('-scf', help = 'Scaling factor for the rotation loss terms', \
	type = float, default = 100)
parser.add_argument('-gamma', help = 'For L2 regularization', \
	type = float, default = 0.0)

##experiments, visualization
parser.add_argument('-expID', help = 'experiment ID', default = 'tmp')
parser.add_argument('-snapshot', help = 'when to take model snapshots', type = int, default = 5)
parser.add_argument('-snapshotStrategy', help = 'Strategy to save snapshots. Note that this has \
	precedence over the -snapshot argument. 1. none: no snapshot at all | 2. default: as frequently \
	as specified in -snapshot | 3. best: keep only the best performing model thus far, \
	4. recent: keep only the most recent model', type = str.lower, \
	choices = ['none', 'default', 'best', 'recent'], default='best')
parser.add_argument('-tensorboardX', help = 'Whether or not to use tensorboardX for \
	visualization', type = bool, default = True)

########### Debugging, Profiling, etc. #######################
parser.add_argument('-debug', help = 'Run in debug mode, and execute 3 quick iterations per train \
	loop. Used in quickly testing whether the code has a silly bug.', type = bool, default = False)
parser.add_argument('-profileGPUUsage', help = 'Profiles GPU memory usage and prints it every \
	train/val batch', type = bool, default = False)
parser.add_argument('-sbatch', help = 'Replaces tqdm and print operations with file writes when \
	True. Useful for reducing I/O when not running in interactive mode (eg. on clusters)', type = bool)



arguments = parser.parse_args()
