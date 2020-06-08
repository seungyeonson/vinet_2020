import args
import csv
import os
import pandas as pd

arg = args.arguments

def write_from_csv_file(path, file_name, output_lines=None) :
    if output_lines is None : output_lines = []
    df = pd.read_csv(path + '/' + file_name)
    if "Unnamed: 0" in (df.columns) :
        cols = df.columns.tolist()
        cols[0] = "index"
        df.columns = cols
    output_lines.append("### {}".format(file_name))
    output_lines.append("columns : {}".format(df.columns.tolist()))
    output_lines.append("col_example : {}".format(df.iloc[0].tolist()))
    output_lines.append("shape : {}".format(df.shape))
    output_lines.append("startFrame : {}".format(df.iloc[0,0]))
    output_lines.append("endFrame : {}".format(df.iloc[-1,0]))
    output_lines.append("\n")
    return output_lines

def write_from_txt_file(path, file_name, output_lines=None) :
    if output_lines is None : output_lines = []
    with open(path + '/' + file_name, 'r') as f :
        lines = f.readlines()
        header = lines[0]
        lines = lines[1:]
    output_lines.append("### {}".format(file_name))
    output_lines.append("columns : {}".format('[' + header.strip().replace(' ', ', '))+ ']')
    output_lines.append("col_example : {}".format(lines[0].strip()))
    output_lines.append("shape : {}".format((len(lines), len(lines[0]))))
    output_lines.append("startFrame : {}".format(lines[0].split(' ')[0]))
    output_lines.append("endFrame : {}".format(lines[-1].split(' ')[0]))
    output_lines.append("\n")
    return output_lines


def make_datainfo(data_path=arg.datadir, type_dataset=arg.dataset) :
    POSE_PATH = os.path.join(data_path, type_dataset, 'poses')
    IMAGE_PATH = os.path.join(data_path, type_dataset, 'images')
    seqs = os.listdir(POSE_PATH)
    for seq in seqs:
        pose_path = os.path.join(POSE_PATH, seq)
        seq_name = os.listdir(pose_path)[0]
        print("{} progress started==============".format(seq + ' ' + seq_name))
        pose_path = os.path.join(pose_path, seq_name)
        image_path = os.path.join(IMAGE_PATH, seq, seq_name)

        output_lines = []
        output_lines = write_from_txt_file(path=image_path, file_name='trimed_left_images.txt', output_lines=output_lines)
        output_lines = write_from_csv_file(path=pose_path, file_name='trimed_imu.csv', output_lines=output_lines)
        output_lines = write_from_txt_file(path=pose_path, file_name='sampled_groundtruth.txt', output_lines=output_lines)
        output_lines = write_from_txt_file(path=pose_path, file_name='sampled_relative_groundtruth.txt', output_lines=output_lines)
        output_lines = write_from_txt_file(path=pose_path, file_name='sampled_relative_R6_groundtruth.txt', output_lines=output_lines)

        with open(pose_path + '/use_data_info.txt', 'w') as f :
            f.writelines([row+'\n' for row in output_lines])

class DataInfo :
    # Create datasets for the current epoch
    # train_seq = [0, 1, 2, 8, 9]
    # train_startFrames = [info_data[seq]['start'] for seq in train_seq]
    # train_endFrames = [info_data[seq]['end'] for seq in train_seq]
    # val_seq = [3, 4, 5, 6, 7, 10]
    # val_startFrames = [0, 0, 0, 0, 0, 0]
    # val_endFrames = [800, 270, 2760, 1100, 1100, 1200]
    def __init__(self):
        self.info_dict = {}
        self.arg = args.arguments
        self.data_path = arg.datadir
        self.type_dataset = arg.dataset
        POSE_PATH = os.path.join(self.data_path, self.type_dataset, 'poses')
        seqs = os.listdir(POSE_PATH)
        for seq in seqs:
            self.info_dict[int(seq)] = {}
            pose_path = os.path.join(POSE_PATH, seq)
            seq_name = os.listdir(pose_path)[0]
            pose_path = os.path.join(pose_path, seq_name)
            with open(pose_path + '/use_data_info.txt', 'r') as f :
                start, end = f.readlines()[4:6]
            start = int(start.replace('startFrame : ', ''))
            end = int(end.replace('endFrame : ', ''))
            self.info_dict[int(seq)]['start'] = start
            self.info_dict[int(seq)]['end'] = end

    def __getitem__(self, item):
        return self.info_dict[item]

    def get_startFrames(self, seqs):
        return [self.info_dict[seq]['start'] for seq in seqs]

    def get_endFrames(self, seqs):
        return [self.info_dict[seq]['end'] for seq in seqs]


if __name__ == "__main__" :
    # make_datainfo(type_dataset='euroc')



    info_dict = DataInfo()
    train_seq = [0, 1, 2, 8, 9]
    train_startFrames = info_dict.get_startFrames(train_seq)
    train_endFrames = info_dict.get_endFrames(train_seq)
    val_seq = [3, 4, 5, 6, 7, 10]
    val_startFrames = info_dict.get_startFrames(val_seq)
    val_endFrames =  info_dict.get_endFrames(val_seq)
    print('train')
    print(train_startFrames)
    print(train_endFrames)

    print('val')
    print(val_startFrames)
    print(val_endFrames)

    pass