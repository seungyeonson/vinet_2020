import helpers
# from lieFunctions import rotMat_to_axisAngle, rotMat_to_quat, rotMat_to_euler
import numpy as np
import os
import scipy.misc as smc
from skimage import io
import csv
import torch
from torch.utils.data import Dataset

import args
#Parse arguments
arg = args.arguments

# Class for providing an iterator for the KITTI visual odometry dataset
class Dataloader(Dataset):

	#constructor
	def __init__(self, BaseDir, sequences=None, startFrames=None, endFrames=None, \
				 width=1280, height=384):
		# Path to base directory of the odometry dataset
		# The base directory contains two directories: 'poses' and 'images'
		# The 'poses' directory contains text files that contain ground-truch pose+imu.txt
		# are present in the 'sequences' folder
		self.baseDir = BaseDir

		# Path to directory containing images
		self.imgDir = os.path.join(self.baseDir, arg.dataset, 'images')
		# Path to directory containing pose ground-truth
		self.poseDir = os.path.join(self.baseDir, arg.dataset, 'poses')

		# Max frames in each dataset sequence

		#TODO: 0 to 10 endframe autonomously & split uzh/euroc by args
		self.MaxFrames = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]

		# Dimensions to be fed in the input
		self.width = width
		self.height = height
		self.channels = 1

		# List of sequences that are part of the dataset
		# If nothing is specified, use sequence 1 as default
		self.sequences = sequences if sequences is not None else list(1)

		# List of start frames and end frames for each sequence
		self.startFrames = startFrames if startFrames is not None else list(0)
		self.endFrames = endFrames if endFrames is not None else list(1100)

		# Variable to hold length of the dataset
		self.len = 0
		# Variables used as caches to implement quick __getitem__ retrieves
		self.cumulativeLengths = [0 for i in range(len(self.sequences))]

		if len(self.sequences) != len(self.startFrames):
			raise ValueError('There are not enough startFrames specified as there are sequences.')
		if len(self.sequences) != len(self.endFrames):
			raise ValueError('There are not enough endFrames specified as there are sequences.')

		for i in range(len(self.sequences)):
			seq = self.sequences[i]
			print(seq)
			print(self.startFrames[i])
			print(self.MaxFrames[seq])
			if self.startFrames[i] < 0 or self.startFrames[i] > self.MaxFrames[seq]:
				raise ValueError('Invalid startFrame for sequence', str(seq).zfill(2))
			if self.endFrames[i] <= 0 or self.endFrames[i] <= self.startFrames[i] or \
					self.endFrames[i] > self.MaxFrames[seq]:
				raise ValueError('Invalid endFrame for sequence', str(seq).zfill(2))
			self.len += (endFrames[i] - startFrames[i])
			self.cumulativeLengths[i] = self.len
		if self.len < 0:
			raise ValueError('Length of the dataset cannot be negative.')

		# Get dataset size
		def __len__(self):

			return self.len

	# __getitem__ method: retrieves an item from the dataset at a specific index
	def __getitem__(self, idx):

		# First determine which sequence the index belongs to, using self.cumulativeLengths
		seqKey = helpers.firstGE(self.cumulativeLengths, idx)
		seqIdx = self.sequences[seqKey]

		# Now select the offset from the first frame of the sequence that the current idx
		# belongs to
		if seqKey == 0:
			offset = idx
		else:
			offset = idx - self.cumulativeLengths[seqKey-1]

		# Map the offset to frame ids
		frame1 = self.startFrames[seqKey] + offset
		frame2 = frame1 + 1

		# Flag to indicate end of sequence
		endOfSequence = False
		if frame2 == self.endFrames[seqKey]:
			endOfSequence = True

		# return (seqIdx, frame1, frame2)

		# Directory containing images for the current sequence
		curImgDir = os.path.join(self.imgDir, str(seqIdx).zfill(2))
		# Read in the corresponding images
		# print(os.path.join(curImgDir, str(frame1).zfill(6) + '.png'))
		# print(os.path.join(curImgDir, str(frame2).zfill(6) + '.png'))
		# print(curImgDir, str(frame1).zfill(6) + '.png')


		# TODO: change variable name 'trim_img'
		trim_img =np.loadtxt(os.path.join(curImgDir,os.listdir(curImgDir)[0], 'learning_data_info.txt'),dtype=str)
		print(frame1,frame2,trim_img[frame1][3] )
		img1 = smc.imread(os.path.join(curImgDir,os.listdir(curImgDir)[0],'left', trim_img[frame1][3]), mode = 'L')
		img2 = smc.imread(os.path.join(curImgDir,os.listdir(curImgDir)[0],'left', trim_img[frame2][3]), mode = 'L')

		img1 = self.preprocessImg(img1)
		img2 = self.preprocessImg(img2)

		# Concatenate the images along the channel dimension (and CUDAfy them)
		pair = torch.empty([1, 2*self.channels, self.height, self.width])

		pair[0] = torch.cat((img1, img2), 0)
		inputTensor = (pair.float()).cuda()
		inputTensor = inputTensor * torch.from_numpy(np.asarray([1. / 255.], \
																dtype = np.float32)).cuda()
		# Load pose ground-truth
		curposeDir = os.path.join(self.poseDir, str(seqIdx).zfill(2))

		poses = np.loadtxt(os.path.join(curposeDir, os.listdir(curposeDir)[0], 'sampled_groundtruth.txt'), \
							   dtype=np.float32)
		relative_R6 = np.loadtxt(os.path.join(curposeDir, os.listdir(curposeDir)[0], 'sampled_relative_R6_groundtruth.txt'), \
							   dtype=np.float32)

		pose1 = np.vstack([poses[frame1].astype(np.float32)])
		#pose1.shape = (1,8) [[timestamp,tx, ty, tz, qx, qy, qz, qw]]


		#relative R6 gt
		pose2 =np.vstack([relative_R6[frame1].astype(np.float32)])
		pose2 = pose2[:,1:]
		# pose2.shape = (1,7) [[r1, r2, r3, r4, r5, r6]]

		# TODO: change imu_index from learning_data_info.txt 's column
		imu_index_1 = trim_img[frame1][0] #!!!!!!!!!!!!!!!!!!!!!!!!!have to change
		imu_index_2 = trim_img[frame2][0] #!!!!!!!!!!!!!!!!!!!!!!!!!have to change

		imu_data = np.loadtxt(os.path.join(curposeDir, os.listdir(curposeDir)[0], 'trimed_imu.txt'), \
							   dtype=np.float32, comments='#')
		imu_data = imu_data[:,1:]
		frame1_imu =imu_data[int(imu_index_1):int(imu_index_2)+1]
		frame2_imu =imu_data[int(imu_index_2)]

		imu = np.resize(frame1_imu, (1,len(frame1_imu),6))


		imu = torch.from_numpy(imu).type(torch.FloatTensor).cuda()

		return inputTensor, pose2, imu, seqIdx, frame1, frame2, endOfSequence

	def preprocessImg(self, img):

		# # Subtract the mean R,G,B pixels
		# img[:,:,0] = (img[:,:,0] - self.channelwiseMean[0])/(self.channelwiseStdDev[0])
		# img[:,:,1] = (img[:,:,1] - self.channelwiseMean[1])/(self.channelwiseStdDev[1])
		# img[:,:,2] = (img[:,:,2] - self.channelwiseMean[2])/(self.channelwiseStdDev[2])
		#
		# Resize to the dimensions required
		img = np.resize(img, (self.height, self.width, self.channels))

		# Torch expects NCWH
		img = torch.from_numpy(img)
		img = img.permute(2,0,1)

		return img


if __name__ == '__main__' :
	pass
	# train_data = Dataloader(arg.datadir, [0,1], [0,0], [4540,1100], width = arg.imageWidth, height = arg.imageHeight)
	# ##error
	# train_data = Dataloader(arg.datadir, [1,0], [0,0], [1100,4540], width = arg.imageWidth, height = arg.imageHeight)
	#
	# print(train_data.len)
	# for i in range(len(train_data):
	# 	train_data[i]

	#if fast test
	# train_data[1]
