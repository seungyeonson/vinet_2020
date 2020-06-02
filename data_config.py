import os
import args

arg = args.arguments

def change_euroc_image_name(data_path=arg.datadir + '/' + arg.dataset + '/images/') :
    '''
    Convert name of EuRoC imagefiles('439388593349.png'(Timestamp)) to 6-digit index('000001.png')
    And, modify information text of images ('left_images.csv') like below :
        ,   #timestamp [ns],      filename,                 newname
        0,  1403638127245096960,  1403638127245096960.png,  000000.png

    Example for use :
        change_euroc_image_name("/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/euroc/images/")
    '''

    import pandas as pd

    DATA_PATH = data_path
    seqs = os.listdir(DATA_PATH)# ['03', '08', etc]
    for seq in seqs:
        path = os.path.join(DATA_PATH, seq)                 # ...../euroc/images/03/
        path = os.path.join(path, os.listdir(path)[0])      # ...../euroc/images/03/V2_01_easy
        for dir in ['left','right'] :
            image_path = os.path.join(path, dir)
            desc_path = os.path.join(path, dir + '_image.csv')
            new_desc_path = os.path.join(path, dir + '_images.csv')

            ## modify csv file
            df = pd.read_csv(desc_path)
            newnames = ['{0:06d}.png'.format(i) for i in range(0, df.shape[0])]
            df['newname'] = newnames
            df.to_csv(desc_path)
            os.rename(desc_path, new_desc_path)

            ## rename images
            for orig, new in zip(df['filename'], newnames) :
                os.rename(image_path + '/' + orig, image_path + '/' + new)


def change_uzh_image_name(data_path=arg.datadir + '/' + arg.dataset + '/images/') :
    '''
    Convert name of UZH imagefiles('1.png') to 6-digit index('000001.png')
    And, modify information text of images ('left_images.csv') like below :
        #   id  timestamp                   image_name(overwrited)
            0   1540824274.074749231339     000001.png

    Example for use :
        change_uzh_image_name("/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/uzh/images/")
    '''

    DATA_PATH = data_path
    seqs = os.listdir(DATA_PATH)  # ['08', '15', etc]
    for seq in seqs:
        path = os.path.join(DATA_PATH, seq)              # ...../uzh/images/08/
        path = os.path.join(path, os.listdir(path)[0])   # ...../uzh/images/08/indoor_forward_12_snapdragon
        for dir in ['left', 'right']:
            image_path = os.path.join(path, dir)
            desc_path = os.path.join(path, dir + '_images.txt')
            image_names = sorted(list(map(lambda x: int(x[:-4]), os.listdir(image_path))))

            ## rename images
            for orig in image_names:
                os.rename(image_path + '/' + str(orig) + '.png', image_path + '/{0:06d}.png'.format(orig))

            ## modify txt file
            orig_txt = None
            with open(desc_path, 'r') as f :
                lines = f.readlines()
                orig_txt = lines.copy()
            with open(desc_path, 'w') as f :
                f.write(lines[0])
                for i in range(1, len(lines)) :
                    row = lines[i].split(' ')
                    idx = int(row[0])
                    row[2] = '{0:06d}.png'.format(image_names[idx])
                    f.write(' '.join(row) + '\n')

def parse_uzh_calib(data_path=arg.datadir + '/' + arg.dataset + '/calibrations/'):
    '''
    This function parse and convert calibration file of UZH dataset,
    to use transform_trajectory, to apply calibration transfromation.

    Example for use :
        parse_uzh_calib('/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/uzh/calibrations/')
    '''

    DATA_PATH = data_path
    types = os.listdir(DATA_PATH)

    for t in types:
        target_file = "results-imucam-..{}_imu.txt".format(t)
        lines_calib = None
        with open(DATA_PATH + t + "/" + target_file, 'r') as f:
            lines = f.readlines()
            indicator = "T_ci:  (imu0 to cam0): "

            ## find indicator
            isBreaked = False
            for idx in range(len(lines)):
                if lines[idx].strip() == indicator.strip():
                    idx += 1
                    isBreaked = True
                    break
            assert (isBreaked)

            # idx = lines.index(indicator) + 1
            lines_calib = lines[idx:idx + 4]
            for i in range(4):
                lines_calib[i] = lines_calib[i].replace('[', '').replace(']', '')
                lines_calib[i] = ",".join(lines_calib[i].split())

        with open(DATA_PATH + t + '/calibration.csv', 'w')as f:
            f.write("\n".join(lines_calib))

if __name__ == '__main__' :
    ## 폴더의 모든 이미지 처리함.
    # change_euroc_image_name("/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/euroc/images/")
    # change_uzh_image_name("/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/uzh/images/")
    change_uzh_image_name(arg.datadir + '/' + 'uzh' + '/images/')
    change_euroc_image_name(arg.datadir + '/' + 'euroc' + '/images/')

    ## result 파일에서 칼리브레이션 알아서 빼주고 그 폴더에 calibration.csv로 저장.
    # parse_uzh_calib('/home/mongsil/workspace/datasets/NewDatasetFormat/dataset/uzh/calibrations/')
    parse_uzh_calib(arg.datadir + '/' + arg.dataset + '/calibrations/')