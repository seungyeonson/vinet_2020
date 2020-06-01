import os
import args

arg = args.arguments

def ssss():
    if arg.dataset == 'uzh':
        seq_frame = {'00': ['000', '002551'],
                     '01': ['000', '004161'],
                     '02': ['000', '001969'],
                     '03': ['000', '003176'],
                     '04': ['000', '004547'],
                     '05': ['000', '002067'],
                     '06': ['000', '002125'],
                     '07': ['000', '002207'],
                     '08': ['000', '001650'],
                     '09': ['000', '002646'],
                     '10': ['000', '001964'],
                     '11': ['000', '002074'],
                     '12': ['000', '001801'],
                     '13': ['000', '001983'],
                     '14': ['000', '001435'],
                     '15': ['000', '001983'],
                     '16': ['000', '001739'],
                     '17': ['000', '001835'],
                     '18': ['000', '000000'],
                     '19': ['000', '002389'],
                     '20': ['000', '002368'],
                     '21': ['000', '003486'],
                     '22': ['000', '002456'],
                     '23': ['000', '002151'],
                     '24': ['000', '002363'],
                     '25': ['000', '002915'],
                     '26': ['000', '002181'],
                     '27': ['000', '001910']

                     }
    for dir_id, img_ids in seq_frame.items():
        dir_path = '{}{}/'.format(par.image_dir, dir_id)
        if not os.path.exists(dir_path):
            continue

        print('reforming_name {} directory'.format(dir_id))
        start, end = img_ids
        start, end = int(start), int(end)
        for idx in range(0, start):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            new_img_name = '{:06d}.png'.format(idx)
            new_img_path = '{}{}/{}'.format(par.image_dir, dir_id, new_img_name)
            if os.path.isfile(img_path):
                os.rename(img_path, new_img_path)
        for idx in range(start, end + 1):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            new_img_name = '{:06d}.png'.format(idx)
            new_img_path = '{}{}/{}'.format(par.image_dir, dir_id, new_img_name)

            if os.path.isfile(img_path):
                print(new_img_path)
                os.rename(img_path, new_img_path)