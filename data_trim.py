# pick the times which every data(image, imu, gt) has.
import os
import args
import csv
arg = args.arguments

def trim_image_imu(data_path=arg.datadir, type_dataset=arg.dataset) :
    print("#### {} datset / Trim Image and IMU timestamps ####".format(type_dataset))
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        print("{} progress started==============".format(seq+' '+seq_name))
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        with open(pose_path + '/imu.csv', 'r') as f:
            imus = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_imu = imus[0]
            imus = imus[1:]
        imu_timestamps = [float(row[0]) for row in imus]

        with open(image_path + '/left_images.csv', 'r') as f:
            images = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_image = images[0]
            images = images[1:]
        image_timestamps = [float(row[1]) for row in images]

        ## check which one is longer
        isLonger_image_first = image_timestamps[0] < imu_timestamps[0]
        isLonger_image_last = image_timestamps[-1] > imu_timestamps[-1]

        ## set the first and last timestamp
        first_timestamp = imu_timestamps[0] if isLonger_image_first else imu_timestamps[0]
        last_timestamp = imu_timestamps[-1] if isLonger_image_last else image_timestamps[-1]

        ## cut the longer part by first and last timestamp
        target_trim = image_timestamps if isLonger_image_first else imu_timestamps
        for i in range(len(target_trim)) :
            if target_trim[i] >= first_timestamp :
                if isLonger_image_first :
                    print('Image has been cutted at FIRST : {} -> {}'.format(len(target_trim), len(target_trim) - i))
                    images = images[i:]
                    image_timestamps = [float(row[1]) for row in images]
                else :
                    print('IMU has been cutted at FIRST : {} -> {}'.format(len(target_trim), len(target_trim) - i))
                    imus = imus[i:]
                    imu_timestamps = [float(row[0]) for row in imus]
                break
                target_trim = image_timestamps if isLonger_image_first else imu_timestamps

        target_trim = image_timestamps if isLonger_image_last else imu_timestamps
        for i in range(len(target_trim)) :
            # print(target_trim[i], last_timestamp)
            if target_trim[i] >= last_timestamp :
                if isLonger_image_last :
                    print('Image has been cutted at LAST : {} -> {}'.format(len(target_trim), i))
                    images = images[:i]
                else :
                    print('IMU has been cutted at LAST : {} -> {}'.format(len(target_trim), i))
                    imus = imus[:i]
                break

        with open(pose_path + '/trimed_imu.csv', 'w') as f:
            f.write(','.join(header_imu) + '\n')
            for row in imus:
                f.write(','.join(row) + '\n')
            print('IMU length : ({})  saved'.format(len(imus)))

        with open(image_path + '/trimed_left_images.csv', 'w') as f:
            f.write(','.join(header_image) + '\n')
            for row in images:
                f.write(','.join(row) + '\n')
            print('Images length : ({})  saved'.format(len(images)))

def trim_image_imu_gt(data_path=arg.datadir, type_dataset=arg.dataset) :
    print("#### {} datset / Trim Image and GT timestamps ####".format(type_dataset))
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        print("{} progress started==============".format(seq+' '+seq_name))
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        with open(pose_path + '/imu.csv', 'r') as f:
            imus = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_imu = imus[0]
            imus = imus[1:]
        imu_timestamps = [int(row[0]) for row in imus]

        with open(pose_path + '/groundtruth_state_groundtruth_estimate.csv', 'r') as f:
            gts = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_gt = gts[0]
            gts = gts[1:]
        gt_timestamps = [int(row[0]) for row in gts]

        with open(image_path + '/left_images.csv', 'r') as f:
            images = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_image = images[0]
            images = images[1:]
        image_timestamps = [int(row[1]) for row in images]

        ## set the first and last timestamp
        first_timestamp = max([gt_timestamps[0], image_timestamps[0], imu_timestamps[0]])
        last_timestamp = min([gt_timestamps[-1], image_timestamps[-1], imu_timestamps[-1]])

        ## cut the longer data by first_timestamp
        if first_timestamp == gt_timestamps[0] :
            for i in range(len(image_timestamps)):
                if image_timestamps[i] >= first_timestamp:
                    print('Image has been cutted at FIRST : {} -> {}'.format(len(image_timestamps), len(image_timestamps) - i))
                    images = images[i:]
                    image_timestamps = [float(row[1]) for row in images]
                    break
            for i in range(len(imu_timestamps)) :
                if imu_timestamps[i] >= first_timestamp :
                    print('Imu has been cutted at FIRST : {} -> {}'.format(len(imu_timestamps), len(imu_timestamps) - i))
                    imus = imus[i:]
                    imu_timestamps = [float(row[0]) for row in imus]
                    break
        elif first_timestamp == image_timestamps[0] :
            for i in range(len(imu_timestamps)) :
                if imu_timestamps[i] >= first_timestamp :
                    print('Imu has been cutted at FIRST : {} -> {}'.format(len(imu_timestamps), len(imu_timestamps) - i))
                    imus = imus[i:]
                    imu_timestamps = [float(row[0]) for row in imus]
                    break
            for i in range(len(gt_timestamps)) :
                if gt_timestamps[i] >= first_timestamp :
                    print('GT has been cutted at FIRST : {} -> {}'.format(len(gt_timestamps), len(gt_timestamps) - i))
                    gts = gts[i:]
                    gt_timestamps = [float(row[0]) for row in gts]
                    break
        elif first_timestamp == imu_timestamps[0] :
            for i in range(len(image_timestamps)):
                if image_timestamps[i] >= first_timestamp:
                    print('Image has been cutted at FIRST : {} -> {}'.format(len(image_timestamps), len(image_timestamps) - i))
                    images = images[i:]
                    image_timestamps = [float(row[1]) for row in images]
                    break
            for i in range(len(gt_timestamps)) :
                if gt_timestamps[i] >= first_timestamp :
                    print('GT has been cutted at FIRST : {} -> {}'.format(len(gt_timestamps), len(gt_timestamps) - i))
                    gts = gts[i:]
                    gt_timestamps = [float(row[0]) for row in gts]
                    break

        ## cut the longer part by last timestamp
        if last_timestamp == gt_timestamps[-1] :
            for i in range(len(image_timestamps)):
                if image_timestamps[i] >= last_timestamp:
                    print('Image has been cutted at Last : {} -> {}'.format(len(image_timestamps), i))
                    images = images[:i]
                    break
            for i in range(len(imu_timestamps)) :
                if imu_timestamps[i] >= last_timestamp :
                    print('Imu has been cutted at Last : {} -> {}'.format(len(imu_timestamps), i))
                    imus = imus[:i]
                    break
        elif last_timestamp == image_timestamps[-1] :
            for i in range(len(imu_timestamps)) :
                if imu_timestamps[i] >= last_timestamp :
                    print('Image has been cutted at Last : {} -> {}'.format(len(imu_timestamps), i))
                    imus = imus[:i]
                    break
            for i in range(len(gt_timestamps)) :
                if gt_timestamps[i] >= last_timestamp :
                    print('GT has been cutted at Last : {} -> {}'.format(len(gt_timestamps), i))
                    gts = gts[:i]
                    break
        elif last_timestamp == imu_timestamps[-1] :
            for i in range(len(image_timestamps)):
                if image_timestamps[i] >= last_timestamp:
                    print('Image has been cutted at Last : {} -> {}'.format(len(image_timestamps), i))
                    images = images[:i]
                    break
            for i in range(len(gt_timestamps)) :
                if gt_timestamps[i] >= last_timestamp :
                    print('GT has been cutted at Last : {} -> {}'.format(len(gt_timestamps), i))
                    gts = gts[:i]
                    break

        with open(pose_path + '/trimed_imu.csv', 'w') as f:
            f.write(','.join(header_imu) + '\n')
            for row in imus:
                f.write(','.join(row) + '\n')
            print('IMU length : ({})  saved'.format(len(imus)))

        with open(pose_path + '/trimed_groundtruth.csv', 'w') as f:
            f.write(','.join(header_gt) + '\n')
            for row in gts:
                f.write(','.join(row) + '\n')
            print('GT length : ({})  saved'.format(len(gts)))

        with open(image_path + '/trimed_left_images.csv', 'w') as f:
            f.write(','.join(header_image) + '\n')
            for row in images:
                f.write(','.join(row) + '\n')
            print('Images length : ({})  saved'.format(len(images)))

if __name__ == "__main__" :
    trim_image_imu_gt(type_dataset='euroc')
    # trim_image_imu(type_dataset='euroc')
    #TODO: 시퀀스 별로 GT가 있는 것과 없는 것의 처리를 다르게 잘 해줘야함 유지에이치 할 때 꼭 필요 !!
