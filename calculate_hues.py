from PIL import Image, ImageSequence
from collections import Counter
import os
import numpy as np
from multiprocessing import Pool

IMG_DIR = 'images'


def tabulate_hues(fn):
    try:
        im = Image.open(fn)
        hsv_im = im.convert('HSV')

        hue_counter = Counter()

        if '.gif' in fn:
            for frame in ImageSequence.Iterator(hsv_im):
                hues = np.asarray(frame)[:,:,0].flatten()
                sats = np.asarray(frame)[:,:,1].flatten()
                hue_counter.update([
                    h for h,s in zip(hues, sats) if s > 0
                ])
        else:
            hues = np.asarray(hsv_im)[:,:,0].flatten()
            sats = np.asarray(hsv_im)[:,:,1].flatten()
            hue_counter.update([
                h for h,s in zip(hues, sats) if s > 0
            ])
    except Exception as e:
        return None

    return hue_counter

def main():
    fn_list = [
        os.path.join(IMG_DIR, fn)
        for fn in  os.listdir(IMG_DIR)
    ]
    hue_dict = dict()

    with Pool(9) as p:
        hue_maps = p.map(tabulate_hues, fn_list)

        
    hue_dict = {
        k: v for k, v in
        zip(fn_list, hue_maps)
        if v is not None
    }

    output_fn = 'hues.csv'

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

                               
if __name__=='__main__':
    main()
