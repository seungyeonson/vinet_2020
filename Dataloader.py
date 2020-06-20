import helpers
# from lieFunctions import rotMat_to_axisAngle, rotMat_to_quat, rotMat_to_euler
import numpy as np
import os
import scipy.misc as smc
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from skimage import io
import csv
import torch
from torch.utils.data import Dataset
from data_info import DataInfo
import args
#Parse arguments
arg = args.arguments
from torch.autograd import Variable
import gc

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
		#TODO: 0 to 10 endframe autonomously & split uzh/euroc by args.
		self.euroc_MaxFrames = [3638, 2999, 2631, 1976, 2221, 2871, 1670, 2093, 2240, 2309, 1890]
		if arg.dataset == 'euroc' : self.MaxFrames = self.euroc_MaxFrames
		elif arg.dataset == 'uzh' : pass

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
			# print('seq',seq)
			# print(self.startFrames[i])
			# print('Max',self.MaxFrames[seq])
			if self.startFrames[i] < 0 or self.startFrames[i] > self.MaxFrames[seq]:
				raise ValueError('Invalid startFrame for sequence', str(seq).zfill(2))
			if self.endFrames[i] <= 0 or self.endFrames[i] <= self.startFrames[i] or \
					(self.endFrames[i]-self.startFrames[i]) > self.MaxFrames[seq]:
				# print(self.endFrames[i],self.MaxFrames[seq])
				raise ValueError('Invalid endFrame for sequence', str(seq).zfill(2))
			self.len += (endFrames[i] - startFrames[i])
			self.cumulativeLengths[i] = self.len
		# print(self.sequences)
		# print(self.endFrames)
		# print(self.startFrames)
		# print('cumulativelength :',self.cumulativeLengths)
		# print('len :', self.len)
		if self.len < 0:
			raise ValueError('Length of the dataset cannot be negative.')

		# load all txt files which are gt, relative_gt, imu, learning_data.
		self.poses_dict = {}
		self.relative_R6_dict = {}
		self.imu_data_dict = {}
		self.data_info_dict = {}

		seq_dir = os.listdir(self.poseDir)
		for seq in seq_dir :
			# self.poses_dict.setdefault(seq)
			# self.relative_R6_dict.setdefault(seq)
			# self.imu_data_dict.setdefault(seq)
			# self.data_info_dict.setdefault(seq)
			#
			curPoseDir = os.path.join(self.poseDir, seq)
			curImgDir = os.path.join(self.imgDir, seq)

			seq = int(seq)
			self.imu_data_dict[seq] = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'trimed_imu.txt'), dtype=np.float32, comments='#')
			self.poses_dict[seq] = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'sampled_groundtruth.txt'), dtype=np.float32)
			self.relative_R6_dict[seq] = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'sampled_relative_R6_groundtruth.txt'), dtype=np.float32)
			self.data_info_dict[seq] = np.loadtxt(os.path.join(curImgDir, os.listdir(curImgDir)[0], 'learning_data.txt'), dtype=str)


	# Get dataset size
	def __len__(self):

		return self.len

	# __getitem__ method: retrieves an item from the dataset at a specific index
	def __getitem__(self, idx):

		# First determine which sequence the index belongs to, using self.cumulativeLengths
		seqKey = helpers.firstGE(self.cumulativeLengths, idx)
		seqIdx = self.sequences[seqKey]
		# print('seqKey, seqIdx :', seqKey,seqIdx)
		# Now select the offset from the first frame of the sequence that the current idx
		# belongs to
		if seqKey == 0:
			offset = idx
		else:
			offset = idx - self.cumulativeLengths[seqKey-1]

		# print('offset :', offset)
		# Map the offset to frame ids
		frame1 = offset
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
		# data_info = np.loadtxt(os.path.join(curImgDir,os.listdir(curImgDir)[0], 'learning_data.txt'),dtype=str)
		data_info = self.data_info_dict[seqIdx]
		# print(frame1,frame2,data_info[frame1][3] )
		img1 = smc.imread(os.path.join(curImgDir,os.listdir(curImgDir)[0],'left', data_info[frame1][3]),mode='L')
		img2 = smc.imread(os.path.join(curImgDir,os.listdir(curImgDir)[0],'left', data_info[frame2][3]),mode='L')
		# print('seq :', seqIdx, 'fraim 1 :',frame1,data_info[frame1][3],'  frame 2 :',frame2, data_info[frame2][3])
		# print(frame2, data_info[frame2][3])
		img1 = self.preprocessImg(img1)
		img2 = self.preprocessImg(img2)
		timestamp = float(data_info[frame1][1])
		# Concatenate the images along the channel dimension (and CUDAfy them)
		pair = torch.empty([1, 2*self.channels, self.height, self.width])

		pair[0] = torch.cat((img1, img2), 0)
		# gc.collect()
		inputTensor = (pair.float()).cuda()
		inputTensor = inputTensor * torch.from_numpy(np.asarray([1. / 255.], \
																dtype = np.float32)).cuda()

		# Load pose ground-truth
		curPoseDir = os.path.join(self.poseDir, str(seqIdx).zfill(2))

		# poses = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'sampled_groundtruth.txt'), \
		# 					   dtype=np.float32)
		# relative_R6 = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'sampled_relative_R6_groundtruth.txt'), \
		# 					   dtype=np.float32)
		poses = self.poses_dict[seqIdx]
		relative_R6 = self.relative_R6_dict[seqIdx]

		#relative R6 gt
		pose1 =np.vstack([relative_R6[frame1].astype(np.float32)])
		pose1 = pose1[:,1:]
		pose1 = np.resize(pose1, (1, 1, 6))
		pose1 = torch.from_numpy(pose1).type(torch.FloatTensor).cuda()
		# pose1.shape = (1,6) [[r1, r2, r3, r4, r5, r6]]

		pose2 = np.vstack([poses[frame1].astype(np.float32)])
		pose2 = pose2[:,1:]
		pose2 = np.resize(pose2, (1, 1, 7))
		pose2 = torch.from_numpy(pose2).type(torch.FloatTensor).cuda()
		#pose2.shape = (1,7) [[tx, ty, tz, qx, qy, qz, qw]]

		# TODO: change imu_index from learning_data_info.txt 's column
		imu_index_1 = data_info[frame1][4] # 4 is imu column index in images/.../learning_data_info.txt
		imu_index_2 = data_info[frame2][4]

		# imu_data = np.loadtxt(os.path.join(curPoseDir, os.listdir(curPoseDir)[0], 'trimed_imu.txt'), \
		# 					   dtype=np.float32, comments='#')
		imu_data = self.imu_data_dict[seqIdx]
		imu_data = imu_data[:,1:]
		# frame1_imu =imu_data[int(imu_index_1):int(imu_index_2)+1]
		frame1_imu =imu_data[int(imu_index_1):int(imu_index_1)+11]

		imu = np.resize(frame1_imu, (1,len(frame1_imu),6))
		imu = torch.from_numpy(imu).type(torch.FloatTensor).cuda()
		# print('Pose :',pose2.shape)
		return inputTensor, imu, pose1, pose2, seqIdx, frame1, frame2,timestamp, endOfSequence

	def preprocessImg(self, img):

		# # Subtract the mean R,G,B pixels
		# img[:,:,0] = (img[:,:,0] - self.channelwiseMean[0])/(self.channelwiseStdDev[0])
		# img[:,:,1] = (img[:,:,1] - self.channelwiseMean[1])/(self.channelwiseStdDev[1])
		# img[:,:,2] = (img[:,:,2] - self.channelwiseMean[2])/(self.channelwiseStdDev[2])
		#
		# Resize to the dimensions required
		# print(img)


		# print(img)
		# img.show()
		img = np.resize(img,(self.width, self.height))
		img = np.array(img)
		img = np.expand_dims(img,0)

		# Torch expects NCWH
		try:

			img = torch.from_numpy(img)
			img = img.permute(0,2,1)
		except TypeError:
			print(img)
			print(img.shape)
			raise Exception()

		return img



if __name__ == '__main__' :
	import time
	# Create datasets for the current epoch
	info_dict = DataInfo()
	train_seq = [2, 3, 4, 8, 9]
	# train_seq = range(0, 11)
	train_startFrames = info_dict.get_startFrames(train_seq)
	train_endFrames = info_dict.get_endFrames(train_seq)
	print('train_seq : {}'.format(train_seq))
	print('train_start : {}'.format(train_startFrames))
	print('train_end : {}'.format(train_endFrames))

	train_data = Dataloader(arg.datadir, train_seq, train_startFrames, train_endFrames, width = arg.imageWidth, height = arg.imageHeight)
	for i in range(len(train_data)):

		cur = time.time()

		metadata = np.asarray(train_data[i][7])
		test = np.concatenate((metadata,train_seq), axis=1)
		print(test)
		print(metadata)
		end = time.time()
		print('idx :',i,'   time : %.5f'%(end-cur))

	# if fast test
	# train_data[1]
