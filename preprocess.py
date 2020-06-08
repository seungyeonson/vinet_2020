#ground-truth processing
#TODO:
# 1. GT data sampling. ( near by Image Timestamp )
# 2. make relative_GT
# 3. make R6_GT
import os
import args
import csv
import numpy as np

import sys
# to import sophuspy from local build
# sys.path.append('/home/mongsil/workspace/build_ws_tf114/Sophus/py')
from sympy import Matrix
from utils.sophus import Se3, So3, Quaternion
import quaternion  # pip install numpy numpy-quaternion numba
import decimal
# from pyquaternion import Quaternion

arg = args.arguments

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 15

def float_to_str(f):
    """
    == This function is from HTLife/VInet ==
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

## xyz quaternion ==> se(3)
def normalize(ww, wx, wy, wz):  # make first number positive
    q = [ww, wx, wy, wz]
    ## Find first negative
    idx = -1
    for i in range(len(q)):
        if q[i] < 0:
            idx = i
            break
        elif q[i] > 0:
            break
    # -1 if should not filp, >=0  flipping index
    if idx >= 0:
        ww = ww * -1
        wx = wx * -1
        wy = wy * -1
        wz = wz * -1
    return ww, wx, wy, wz


def xyzQuaternion2se3_(arr):
    # the dataset was like => tx ty tz qx qy qz qw
    x, y, z, wx, wy, wz, ww = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]
    trans = Matrix([x, y, z])
    ww, wx, wy, wz = normalize(ww, wx, wy, wz)

    q_real = ww
    q_img = Matrix([wx, wy, wz])
    q = Quaternion(q_real, q_img)
    # q = Quaternion(ww, wx, wy, wz)
    R = So3(q)

    RT = Se3(R, trans)
    # print(RT.log())
    numpy_vec = np.array(RT.log()).astype(float)  # SE3 to se3

    return np.concatenate(numpy_vec)

def getClosestIndex(searchTime, searchStartIndex, timeList):
    foundIdx = 0
    no_break = False
    for i in range(searchStartIndex, len(timeList)):
        no_break = False
        if timeList[i] >= searchTime:
            foundIdx = i
            break
        no_break = True
    return foundIdx #if not no_break else None

def sampling_GT(data_path=arg.datadir, type_dataset=arg.dataset) :
    if type_dataset == 'uzh' :
        pass
        # POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
        # IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
        # seqs = os.listdir(POSE_PATH)
        # for seq in seqs:
        #     pose_path = os.path.join(POSE_PATH, seq)
        #     seq_name = os.listdir(pose_path)[0]
        #     print(seq_name[-7:])
        #     if seq_name[-7:] != 'with_gt' : continue
        #     pose_path = os.path.join(pose_path, seq_name)
        #     image_path = os.path.join(IMAGE_PATH, seq, seq_name)
        #
        #     gt_timestamps = None
        #     image_timestamps = None
        #     with open(pose_path+'/groundtruth.txt', 'r') as f :
        #         gts = f.readlines()[1:]
        #     gt_timestamps = [float(row.split(' ')[1]) for row in gts]
        #
        #     with open(image_path+'/left_images.txt', 'r') as f :
        #         image_timestamps = f.readlines()[1:]
        #     print(len(image_timestamps))
        #     image_timestamps = [float(row.split(' ')[1]) for row in image_timestamps]
        #
        #     ## trim imu timestamps as image's
        #     for i in range(len(gt_timestamps)) :
        #         if gt_timestamps[i] < image_timestamps[i] :
        #             continue
        #         else:
        #             print("Imu's length has been chaged {} -> {}".format(len(gt_timestamps), len(gt_timestamps)-i))
        #             gt_timestamps = gt_timestamps[i:]
        #             break
        #
        #     sampledGT = []
        #     searchStartIndex = 0
        #     for searchTime in image_timestamps:
        #         foundIdx = getClosestIndex(searchTime, searchStartIndex, gt_timestamps)
        #         if foundIdx is None: continue
        #         sampledGT.append(gts[foundIdx])
        #     for p in sampledGT :
        #         print(p)

    elif type_dataset == 'euroc' :
        from rpg.utils.asl_groundtruth_to_pose import extract   # for converting gt style

        POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
        IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
        seqs = os.listdir(POSE_PATH)
        for seq in seqs:
            pose_path = os.path.join(POSE_PATH, seq)
            seq_name = os.listdir(pose_path)[0]
            pose_path = os.path.join(pose_path, seq_name)
            image_path = os.path.join(IMAGE_PATH, seq, seq_name)

            gt_timestamps = None
            image_timestamps = None
            with open(pose_path + '/trimed_groundtruth.csv', 'r') as f:
                gts = list(csv.reader(f, delimiter=',', quotechar='|'))
                header = gts[0]
                gts = gts[1:]
            gt_timestamps = [row[0] for row in gts]

            with open(image_path + '/trimed_left_images.txt', 'r') as f:
                image_timestamps = list(map(lambda x: x.strip().split(' '), f.readlines()[1:]))
                # image_timestamps = list(csv.reader(f, delimiter=',', quotechar='|'))[1:]
            image_timestamps = [row[1] for row in image_timestamps]

            # ## trim gt timestamps as image's
            # for i in range(len(gt_timestamps)):
            #     if gt_timestamps[i] < image_timestamps[i]:
            #         continue
            #     else:
            #         print("Imu's length has been chaged {} -> {}".format(len(gt_timestamps), len(gt_timestamps) - i))
            #         gt_timestamps = gt_timestamps[i:]
            #         break

            sampledGT = []
            searchStartIndex = 0
            indexs = []
            for searchTime in image_timestamps:
                foundIdx = getClosestIndex(searchTime, searchStartIndex, gt_timestamps)
                if foundIdx is None: continue
                searchStartIndex = foundIdx
                indexs.append(foundIdx)
                sampledGT.append(gts[foundIdx])

            with open(pose_path + '/sampled_groundtruth.csv', 'w') as f :
                f.write(','.join(header) + '\n')
                for row in sampledGT:
                    f.write(','.join(row) + '\n')
            with open(pose_path + '/sampled_groundtruth_index.txt', 'w') as f:
                for row in indexs :
                    f.write(str(row) + '\n')

            ## convert euroc gt file to uzh style
            print('Extract ground truth pose from file ' + pose_path + '/sampled_groundtruth.csv')
            print('Saving to file ' + pose_path + '/sampled_groundtruth.txt')
            extract(pose_path + '/sampled_groundtruth.csv', pose_path + '/sampled_groundtruth.txt')

def relative_GT(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        pose_path = os.path.join(pose_path, seq_name)

        trajectory_abs = []  # abosolute camera pose
        with open(pose_path + '/sampled_groundtruth.txt') as f:
            lines = f.readlines()
            header = lines[0]
            lines = lines[1:]
            for row in list(map(lambda x: x.strip().split(' '), lines)) :
                trajectory_abs.append(row)

        print('Total data: ' + str(len(trajectory_abs)))

        ## Calculate relative pose
        trajectory_relative = []
        for i in range(len(trajectory_abs) - 1):
            # timestamp tx ty tz qx qy qz qw
            timestamp = trajectory_abs[i + 1][0]
            X, Y, Z = np.array(trajectory_abs[i + 1][1:4]).astype(float) - np.array(trajectory_abs[i][1:4]).astype(
                float)

            wx0, wy0, wz0, ww0 = np.array(trajectory_abs[i][4:]).astype(float)
            wx1, wy1, wz1, ww1 = np.array(trajectory_abs[i + 1][4:]).astype(float)
            q0 = np.quaternion(ww0, wx0, wy0, wz0)
            q1 = np.quaternion(ww1, wx1, wy1, wz1)
            relative_rot = quaternion.as_float_array(q1 * q0.inverse())

            relative_pose = [timestamp, X, Y, Z, relative_rot[1], relative_rot[2], relative_rot[3], relative_rot[0]]
            trajectory_relative.append(relative_pose)

        with open(pose_path + '/sampled_relative_groundtruth.txt', 'w') as f:
            # tmpStr = " ".join(trajectory_abs[0])
            f.write(header)

            for i in range(len(trajectory_relative)):
                tmpStr = trajectory_relative[i][0] + ' ' + \
                         float_to_str(trajectory_relative[i][1]) + ' ' + \
                         float_to_str(trajectory_relative[i][2]) + ' ' + \
                         float_to_str(trajectory_relative[i][3]) + ' ' + \
                         float_to_str(trajectory_relative[i][4]) + ' ' + \
                         float_to_str(trajectory_relative[i][5]) + ' ' + \
                         float_to_str(trajectory_relative[i][6]) + ' ' + \
                         float_to_str(trajectory_relative[i][7])
                f.write(tmpStr + '\n')

def r6_GT(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        pose_path = os.path.join(pose_path, seq_name)

        traj_rel = []
        with open(pose_path + '/sampled_relative_groundtruth.txt', 'r') as f:
            lines = f.readlines()
            header = lines[0]
            lines = lines[1:]
            for row in list(map(lambda x: x.strip().split(' '), lines)):
                traj_rel.append(row)
        print("The total length of relative trajectory: ", len(traj_rel), np.array(traj_rel).shape)

        traj_rel_se3R6 = []
        for i in range(len(traj_rel)):
            timestamp = traj_rel[i][0]
            arr = np.array(traj_rel[i][1:]).astype(float)
            se3R6 = xyzQuaternion2se3_(arr)
            traj_rel_se3R6.append(se3R6)

        with open(pose_path + '/sampled_relative_R6_groundtruth.txt', 'w') as f:
            # f.write(header)
            f.write('# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z\n')

            for i in range(len(traj_rel_se3R6)):
                r1 = float_to_str(traj_rel_se3R6[i][0])
                r2 = float_to_str(traj_rel_se3R6[i][1])
                r3 = float_to_str(traj_rel_se3R6[i][2])
                r4 = float_to_str(traj_rel_se3R6[i][3])
                r5 = float_to_str(traj_rel_se3R6[i][4])
                r6 = float_to_str(traj_rel_se3R6[i][5])
                tmpStr = traj_rel[i][0] + ' ' + r1 + ' ' + r2 + ' ' + r3 + ' ' + r4 + ' ' + r5 + ' ' + r6
                f.write(tmpStr + '\n')

        print('The length of traj_rel: ', len(traj_rel), "; The length of traj_rel_se3R6: ", len(traj_rel_se3R6))

def sampling_Imu(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        with open(pose_path + '/trimed_imu.csv', 'r') as f:
            # imus = f.readlines()
            # imus = list(map(lambda x: x.strip().split(' '), imus[1:]))
            imus = list(csv.reader(f, delimiter=',', quotechar='|'))
            header = imus[0]
            imus = imus[1:]
        imu_timestamps = [float(row[0]) for row in imus]

        with open(image_path + '/trimed_left_images.txt', 'r') as f:
            image_timestamps = list(map(lambda x: x.strip().split(' '), f.readlines()))[1:]
        image_timestamps = [float(row[1]) for row in image_timestamps]

        sampledImu = []
        indexs = []
        searchStartIndex = 0
        for searchTime in image_timestamps:
            foundIdx = getClosestIndex(searchTime, searchStartIndex, imu_timestamps)
            if foundIdx is None: continue
            searchStartIndex = foundIdx
            indexs.append(foundIdx)
            sampledImu.append(imus[foundIdx])

        with open(pose_path + '/sampled_imu.txt', 'w') as f:
            f.write(' '.join(header) + '\n')
            for row in sampledImu:
                f.write(' '.join(row) + '\n')

        with open(pose_path + '/sampled_imu_index.txt', 'w') as f:
            for row in indexs:
                f.write(str(row) + '\n')

def prepare_learn_data(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        with open(image_path + '/trimed_left_images.txt', 'r') as f:
            image_data = f.readlines()
            image_data = [row.strip() for row in image_data]
        with open(pose_path + '/sampled_imu.txt', 'r') as f :
            imu_data = f.readlines()
            imu_data = [row.strip().split(' ') for row in imu_data]
        with open(pose_path + '/sampled_imu_index.txt', 'r') as f :
            imu_index = f.readlines()
            imu_index = [row.strip() for row in imu_index]
        with open(pose_path + '/sampled_groundtruth.txt', 'r') as f:
            gt_data = f.readlines()
            gt_data= [row.strip().split(' ') for row in gt_data]
        with open(pose_path + '/sampled_groundtruth_index.txt', 'r') as f :
            gt_index = f.readlines()
            gt_index = [row.strip() for row in gt_index]
        assert len(image_data) == len(imu_data) == len(gt_data)


        all_data = []
        all_data.append('#image_index ' + image_data[0] + ' ' + 'imu_index imu_timestamps gt_index gt_timestamps\n')
        for i in range(1,len(image_data)) :
            all_data.append(image_data[i] + ' ' + imu_index[i-1] + ' ' + imu_data[i][0] + ' ' + gt_index[i-1] + ' ' + gt_data[i][0] + '\n')

        with open(image_path + '/learning_data.txt', 'w') as f :
            f.writelines(all_data)

if __name__ == "__main__" :
    sampling_GT(type_dataset='euroc')
    relative_GT(type_dataset='euroc')
    r6_GT(type_dataset='euroc')
    sampling_Imu(type_dataset='euroc')
    prepare_learn_data(type_dataset='euroc')