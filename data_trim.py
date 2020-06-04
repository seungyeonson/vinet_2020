# pick the times which every data(image, imu, gt) has.
import os
import args
import csv
arg = args.arguments

def sync_image_imu(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        imu_timestamps = None
        image_timestamps = None
        with open(pose_path + '/imu.csv', 'r') as f:
            imus = list(csv.reader(f, delimiter=',', quotechar='|'))
            header = imus[0]
            imus = imus[1:]
        imu_timestamps = [float(row[0]) for row in imus]

        with open(image_path + '/left_images.csv', 'r') as f:
            image_timestamps = list(csv.reader(f, delimiter=',', quotechar='|'))[1:]
        image_timestamps = [float(row[1]) for row in image_timestamps]

        if
        isLonger_image_first = False
        isLonger_image_last = False



if __name__ == "__main__" :
    sync_image_imu(type_dataset='euroc')