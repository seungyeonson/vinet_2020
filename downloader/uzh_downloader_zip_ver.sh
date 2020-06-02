#!/bin/bash

files=(
#00:indoor_forward_3_snapdragon_with_gt.zip
#01:indoor_forward_5_snapdragon_with_gt.zip
#02:indoor_forward_6_snapdragon_with_gt.zip
#03:indoor_forward_7_snapdragon_with_gt.zip
#04:indoor_forward_8_snapdragon.zip
#05:indoor_forward_9_snapdragon_with_gt.zip
#06:indoor_forward_10_snapdragon_with_gt.zip
#07:indoor_forward_11_snapdragon.zip
08:indoor_forward_12_snapdragon.zip
#09:indoor_45_1_snapdragon.zip
#10:indoor_45_2_snapdragon_with_gt.zip
#11:indoor_45_3_snapdragon.zip
12:indoor_45_4_snapdragon_with_gt.zip
#13:indoor_45_9_snapdragon_with_gt.zip
#14:indoor_45_11_snapdragon.zip
#15:indoor_45_12_snapdragon_with_gt.zip
#16:indoor_45_13_snapdragon_with_gt.zip
#17:indoor_45_14_snapdragon_with_gt.zip
#18:indoor_45_16_snapdragon.zip
#19:outdoor_forward_1_snapdragon_with_gt.zip
#20:outdoor_forward_2_snapdragon.zip
#21:outdoor_forward_3_snapdragon_with_gt.zip
#22:outdoor_forward_5_snapdragon_with_gt.zip
#23:outdoor_forward_6_snapdragon.zip
#24:outdoor_forward_9_snapdragon.zip
#25:outdoor_forward_10_snapdragon.zip
#26:outdoor_45_1_snapdragon_with_gt.zip
#27:outdoor_45_2_snapdragon.zip
)

echo 'start UZH downloading......'
mkdir -p 'dataset/uzh/images'
mkdir 'dataset/uzh/poses'
dirImage='dataset/uzh/images/'
dirPose='dataset/uzh/poses/'

#files_len=${#files}
for i in ${files[@]}; do
#for idx in seq 0 $files_len; do
#	i=${files[idx]}
	withGT='no'
	mkdir 'temp'	
	extracted='./temp/'
	if [ ${i:(-11):7} == "with_gt" ]
	then
		shortname=${i:3}
		withGT='yes'
		len=${#shortname}
	        rename=${i:3:$len-4}
		leica=${shortname:0:$len-23}'.zip'
		sequence=${i:0:2}
		
	        #wget 'http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v2/'$shortname
		#wget 'http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/raw/'$leica
	else
		shortname=${i:3}	        
		len=${#shortname}
		rename=${i:3:$len-4}
		sequence=${i:0:2}
		
		withGT='no'
		#wget 'http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v2/'$shortname
	fi

	echo '=======start unzipping......======='
        unzip -o $shortname -d $extracted
#        rm $shortname	
	echo '=======end unzipping======='

	echo '=======start rearanging directories...======='
       
	 ## Remove right images and Move to main folder

#        rm -r $extracted'right_images.txt'
	mkdir -p $dirImage$sequence'/'$rename'/right'
	mkdir -p $dirImage$sequence'/'$rename'/left'
	imageListR=(`ls $extracted'img/' | grep ^'image_1' | sort`)
	for imagesR in ${imageListR[@]}; do
		mv $extracted'img/'$imagesR $dirImage$sequence'/'$rename'/right/'${imagesR:8}
	done
	imageListL=(`ls $extracted'img/' | grep ^'image_0' | sort`)
	for imagesL in ${imageListL[@]}; do
		mv $extracted'img/'$imagesL $dirImage$sequence'/'$rename'/left/'${imagesL:8}
	done


#        mv $extracted'img' $dirImage$sequence'/'$rename
	mkdir -p $dirPose$sequence'/'$rename
	mv $extracted'imu.txt' $dirPose$sequence'/'$rename'/imu.txt'
	mv $extracted'groundtruth.txt' $dirPose$sequence'/'$rename'/groundtruth.txt'
	mv $extracted'left_images.txt' $dirImage$sequence'/'$rename'/left_images.txt'
	mv $extracted'right_images.txt' $dirImage$sequence'/'$rename'/right_images.txt'
        rm -r $extracted

	## Additional touch, if it has GT
	if [ $withGT == 'yes' ]
	then
		unzip -o $leica
		len=${#leica}
		mv ${leica:0:$len-4}'/leica.txt' $dirPose$sequence'/'$rename'/leica.txt'
		rm -r ${leica:0:$len-4}
	fi
echo '=======all downloading done!======='
done

