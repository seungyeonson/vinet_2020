#!/bin/bash

files=(
#00:MH_01_easy.zip
#01:MH_02_easy.zip
#02:MH_03_medium.zip
03:MH_04_difficult.zip
#04:MH_05_difficult.zip
#05:V1_01_easy.zip
#06:V1_02_medium.zip
#07:V1_03_difficult.zip
08:V2_01_easy.zip
#09:V2_02_medium.zip
#10:V2_03_difficult.zip
)

echo '==============start EuRoC downloading......================'
mkdir -p 'dataset/euroc/images'
mkdir 'dataset/euroc/poses'
dirImage='dataset/euroc/images/'
dirPose='dataset/euroc/poses/'

for i in ${files[@]}; do
	mkdir 'temp'	
	extracted='./temp/'

	shortname=${i:3}
	len=${#shortname}
	nonZipName=${i:3:$len-4}
        rename=$nonZipName
	sequence=${i:0:2}
	if [ ${shortname:0:2} == "MH" ]
	then
		roomname='machine_hall'
	elif [ ${shortname:0:2} == "V1" ]
	then
		roomname='vicon_room1'
	elif [ ${shortname:0:2} == "V2" ]
	then
		roomname='vicon_room2'
	fi

        #wget 'http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/'$roomname'/'$nonZipName'/'$shortname


	echo '=======start unzipping......======='
        unzip -o $shortname -d $extracted
#        rm $shortname	
	echo '=======end unzipping======='
	
       
	## Remove right images and Move to main folder
	echo '=======start rearanging directories...======='
	mkdir -p $dirPose$sequence'/'$rename
	mkdir -p $dirImage$sequence'/'$rename'/left'
	mkdir -p $dirImage$sequence'/'$rename'/right'
	imageListL=(`ls $extracted'mav0/cam0/data/' | sort`)
	imageListR=(`ls $extracted'mav0/cam1/data/' | sort`)
	for L in ${imageListL[@]}; do
		mv $extracted'mav0/cam0/data/'$L $dirImage$sequence'/'$rename'/left/'$L
	done
	for R in ${imageListR[@]}; do
		mv $extracted'mav0/cam1/data/'$R $dirImage$sequence'/'$rename'/right/'$R
	done
	mv $extracted'mav0/cam0/data.csv' $dirImage$sequence'/'$rename'/left_images.csv'
	mv $extracted'mav0/cam1/data.csv' $dirImage$sequence'/'$rename'/right_images.csv'


	mv $extracted'mav0/imu0/data.csv' $dirPose$sequence'/'$rename'/imu.csv'
	mv $extracted'mav0/vicon0/data.csv' $dirPose$sequence'/'$rename'/groundtruth_vicon.csv'
	mv $extracted'mav0/leica0/data.csv' $dirPose$sequence'/'$rename'/groundtruth_leica.csv'
	mv $extracted'mav0/state_groundtruth_estimate0/data.csv' $dirPose$sequence'/'$rename'/groundtruth_state_groundtruth_estimate.csv'
        rm -r $extracted

echo '=======all downloading done!======='
done

