from PIL import Image, ImageSequence
from collections import Counter
import os
import numpy as np
import argparse

from multiprocessing import Pool

#IMG_DIR = 'images'


def tabulate_hues(fn):
    try:
        im = Image.open(fn)
        hsv_im = im.convert('HSV')

        hue_counter = Counter()

        if '.gif' in fn:
            for frame in ImageSequence.Iterator(hsv_im):
                npframe = np.asarray(frame)
                hues = npframe[:,:,0].flatten()
                sats = npframe[:,:,1].flatten()
                hue_counter.update([
                    h for h,s in zip(hues, sats) if s > 0
                ])
        else:
            npframe = np.asarray(hsv_im)
            hues = npframe[:,:,0].flatten()
            sats = npframe[:,:,1].flatten()
            hue_counter.update([
                h for h,s in zip(hues, sats) if s > 0
            ])
    except Exception as e:
        return None

    return hue_counter

def main(options):
    fn_list = [
        os.path.join(options['image_directory'], fn)
        for fn in  os.listdir(options['image_directory'])
    ]

    if options['subdirectories']:
        print(fn_list)
        fn_list = [
            os.path.join(subdir, fn)
            for subdir in fn_list
            for fn in os.listdir(subdir)
        ]
    hue_dict = dict()

    print(len(fn_list))

    with Pool(9) as p:
        hue_maps = p.map(tabulate_hues, fn_list)

        
    hue_dict = {
        k: v for k, v in
        zip(fn_list, hue_maps)
        if v is not None
    }

    output_fn = options['outfile']

    with open(output_fn, 'w') as f:
        f.write('fn,hue,cnt\n')
        for k, v in hue_dict.items():
            if v is None:
                continue
            for hue, cnt in v.items():
                f.write('%s,%s,%s\n' % (
                    k, hue, cnt
                ))

    print('Done!')

def get_options():
    parser = argparse.ArgumentParser(
        description='Tabulate hues from images'
    )

    parser.add_argument(
        'image_directory',
        help='Directory to tabulate images from',
    )

    parser.add_argument(
        '--outfile',
        default='hues.csv',
        help='Output file'
    )

    parser.add_argument(
        '--subdirectories',
        action='store_true',
        help='Look at files in subdirectories'
    )

    args = parser.parse_args()
    options = vars(args)
    return options
        
                               
if __name__=='__main__':
    options = get_options()
    main(options)
