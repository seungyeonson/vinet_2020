#!/bin/bash

files=(
outdoor_45_calib_snapdragon
outdoor_forward_calib_snapdragon
indoor_forward_calib_snapdragon
indoor_45_calib_snapdragon
)

echo 'start UZH calibration downloading......'
mkdir -p 'dataset/uzh/calibrations'
dirCalib='dataset/uzh/calibrations'

for name in ${files[@]}; do
	fullname=$name'.zip'

	wget 'http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/'$fullname

	echo '=======start unzipping......======='
        unzip -e $fullname -d $dirCalib'/'
	echo '=======end unzipping======='
	rm $fullnames
       

done
echo '=======all downloading done!======='
