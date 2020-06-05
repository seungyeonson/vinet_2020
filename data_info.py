import args
import csv
import os
arg = args.arguments

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

        with open(image_path + '/trimed_left_images.csv', 'r') as f:
            images = list(csv.reader(f, delimiter=',', quotechar='|'))
            header_image = images[0]
            images = images[1:]

class DataInfo :
    def __init__(self):
        self.arg = args.arguments
        self.data_path = args.datadir


if __name__ == "__main__" :
    pass