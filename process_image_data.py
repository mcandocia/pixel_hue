from PIL import Image, ImageSequence
from collections import Counter
import os
import numpy as np
import argparse
from functools import wraps
from functools import partial
from scipy.stats import circmean, circstd
from copy import deepcopy
from pg_conn import make_conn
import re
import traceback


from multiprocessing import Pool

# FEATURES:
# HEIGHT, WIDTH (2)
# R/G/B averages (3)
# H/S/V averages (3) 
# N_FRAMES (1)
# TYPE FLAG (PNG/JPG/GIF/OTHER) (4)
# N_COLORS (1)

# PCT WHITE (1)
# PCT BLACK (1)
# PCT GRAYSCALE (SAT=0) (1)
# TOTAL: 14

# QUADRANTS
# ALL OF ABOVE x 4
# TOTAL: 56

# GRAND TOTAL: 70


#IMG_DIR = 'images'

def insert_statistics_to_db(
        statistics,
        cur,
        options
):
    # normal statistics

    normal_stats = [
        (k,v) for k, v in statistics.items()
        if k != 'divided_statistics'
    ]

    normal_vals = [x[1] for x in normal_stats]
    normal_keys = ','.join([x[0] for x in normal_stats])
    normal_placeholder = ','.join(['%s' for _ in normal_vals])

    # divided statistics
    #print(statistics['divided_statistics'])

    divided_stats = [
        ('div_%s_%s_%s' % (f[0],f[1],k),v)
        for k, d in 
        statistics['divided_statistics'].items()
        for f, v in d.items()
    ]

    divided_vals = [x[1] for x in divided_stats]
    divided_keys = ','.join([x[0] for x in divided_stats])
    divided_placeholder =  ','.join(['%s' for _ in divided_stats])

    # combine
    all_vals = normal_vals + divided_vals
    all_keys = ',\n'.join([normal_keys, divided_keys])
    all_placeholder = ','.join([normal_placeholder,divided_placeholder])
    
    query = """
    INSERT INTO {tablename}(
     {all_keys}
    ) VALUES ({all_placeholder})
    """.format(
        all_keys=all_keys,
        all_placeholder=all_placeholder,
        tablename=options['data_tablename']
    )

    cur.execute(query, all_vals)

    return 0

    
    


def process_image(fn, subreddit, options, verbose = False):
    reddit_id = re.sub(r'.*/([a-z0-9]+)\..*?$',r'\1', fn)    
    try:
        im = Image.open(fn)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        file_size = os.stat(fn).st_size
        filetype = re.sub(r'.*\.','', fn)


        hsv_im = im.convert('HSV')
        conn = make_conn()
        cur = conn.cursor()

        hue_counter = Counter()

        splitter_decorator = split_by(nx=options['nx'], ny=options['ny'], other_vars='mask')
        os.makedirs(
            os.path.join(
                options['processed_directory'],
                subreddit
            ),
            exist_ok=True
        )

        if '.gif' in fn:
            frame_idx = 0
            for rgb_frame in (
                    ImageSequence.Iterator(im)
                ):
                    hsv_frame = rgb_frame.convert('HSV')
                    np_hue = np.asarray(hsv_frame)
                    np_rgb = np.asarray(rgb_frame)
                    # RGB
                    rcol = np_rgb[:,:,0]
                    gcol = np_rgb[:,:,1]
                    bcol = np_rgb[:,:,2]

                    # HSV
                    hues = np_hue[:,:,0]
                    sats = np_hue[:,:,1]
                    vals = np_hue[:,:,2]

                    sat_mask = sats <= options['saturation_mask_threshold']


                    ## OVERALL STATS
                    # DIMS
                    statistics = {'frame_idx': frame_idx, 'reddit_id': reddit_id, 'subreddit':subreddit}
                    statistics['width'], statistics['height'] = rgb_frame.size                    

                    ## CHANNEL STATS
                    # RGB STATS
                    for key, data in zip(
                            ['red','blue','green'],
                            [rcol,gcol,bcol],
                    ):
                        statistics['%s_avg' % key] = get_avg(data)
                        statistics['%s_std' % key] = get_std(data)

                    # HSV STATS
                    for key, data in zip(
                            ['hue','saturation','value'],
                            [hues,sats,vals],
                    ):
                        if key == 'hue':
                            statistics['%s_avg' % key] = get_avg(data, circular=key=='hue', mask=~sat_mask)
                            statistics['%s_std' % key] = get_std(data, circular=key=='hue', mask=~sat_mask)
                        else:
                            statistics['%s_avg' % key] = get_avg(data, circular=key=='hue')
                            statistics['%s_std' % key] = get_std(data, circular=key=='hue')
                    # NO. COLORS & PCT
                    statistics['n_unique_colors'] = get_unique_colors(np_rgb)
                    statistics['pct_white'] = get_pct_color(np_rgb,(255,255,255))
                    statistics['pct_black'] = get_pct_color(np_rgb,(0,0,0))
                    statistics['pct_grayscale'] = np.mean(sat_mask.flatten())

                    ## DIVIDED CHANNEL STATS
                    divided_statistics = {}                    
                    # RGB STATS
                    for key, data in zip(
                            ['red','blue','green'],
                            [rcol,gcol,bcol],
                    ):
                        divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(data)
                        divided_statistics['%s_std' % key] = splitter_decorator(get_std)(data)                    

                    # HSV STATS
                    for key, data in zip(
                            ['hue','saturation','value'],
                            [hues,sats,vals],
                    ):
                        if key == 'hue':
                            divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(
                                data, circular=key=='hue', mask=~sat_mask
                            )
                            divided_statistics['%s_std' % key] = splitter_decorator(get_std)(
                                data, circular=key=='hue', mask=~sat_mask
                            )
                        else:
                            divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(data, circular=key=='hue')
                            divided_statistics['%s_std' % key] = splitter_decorator(get_std)(data, circular=key=='hue')                        

                    # NO. COLORS                    
                    divided_statistics['n_unique_colors'] = splitter_decorator(get_unique_colors)(np_rgb)
                    divided_statistics['pct_white'] = splitter_decorator(get_pct_color)(np_rgb,(255,255,255))
                    divided_statistics['pct_black'] = splitter_decorator(get_pct_color)(np_rgb,(0,0,0))
                    divided_statistics['pct_grayscale'] = splitter_decorator(lambda x: np.mean(x.flatten()))(sat_mask)
                    

                    # END LOOP
                    statistics['divided_statistics'] = divided_statistics
                    
                    statistics['source_filename'] = fn
                    statistics['target_filename'] = os.path.join(
                        options['processed_directory'],
                        subreddit,
                        '%s_%s.png' % (reddit_id, str(frame_idx).zfill(4))

                    )
                    statistics['filesize'] = file_size
                    statistics['filetype'] = filetype
                    standardized_image = standardize_image(rgb_frame, options)
                    standardized_image.save(statistics['target_filename'])
                    frame_idx += 1

                    insert_statistics_to_db(statistics, cur, options)

        else:
            np_rgb = np.asarray(im)
            np_hue = np.asarray(hsv_im)

            # to match code in GIF loop
            rgb_frame = im
            hsv_frame = hsv_im
            
            # RGB
            rcol = np_rgb[:,:,0]
            gcol = np_rgb[:,:,1]
            bcol = np_rgb[:,:,2]

            # HSV
            hues = np_hue[:,:,0]
            sats = np_hue[:,:,1]
            vals = np_hue[:,:,2]

            sat_mask = sats <= options['saturation_mask_threshold']

            ## OVERALL STATS
            # DIMS
            statistics = {'frame_idx': -1, 'reddit_id': reddit_id, 'subreddit':subreddit}
            statistics['width'], statistics['height'] = rgb_frame.size                    

            ## CHANNEL STATS
            # RGB STATS
            for key, data in zip(
                    ['red','blue','green'],
                    [rcol,gcol,bcol],
            ):
                statistics['%s_avg' % key] = get_avg(data)
                statistics['%s_std' % key] = get_std(data)

            # HSV STATS
            for key, data in zip(
                    ['hue','saturation','value'],
                    [hues,sats,vals],
            ):
                if key == 'hue':
                    statistics['%s_avg' % key] = get_avg(data, circular=key=='hue', mask=~sat_mask)
                    statistics['%s_std' % key] = get_std(data, circular=key=='hue', mask=~sat_mask)
                else:
                    statistics['%s_avg' % key] = get_avg(data, circular=key=='hue')
                    statistics['%s_std' % key] = get_std(data, circular=key=='hue')                    

            # NO. COLORS & PCT
            statistics['n_unique_colors'] = get_unique_colors(np_rgb)
            statistics['pct_white'] = get_pct_color(np_rgb,(255,255,255))
            statistics['pct_black'] = get_pct_color(np_rgb,(0,0,0))
            statistics['pct_grayscale'] = np.mean(sat_mask.flatten())

            ## DIVIDED CHANNEL STATS
            divided_statistics = {}                    
            # RGB STATS
            for key, data in zip(
                    ['red','blue','green'],
                    [rcol,gcol,bcol],
            ):
                divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(data)
                divided_statistics['%s_std' % key] = splitter_decorator(get_std)(data)                    

            # HSV STATS
            for key, data in zip(
                    ['hue','saturation','value'],
                    [hues,sats,vals],
            ):
                if key == 'hue':
                    divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(data, circular=key=='hue', mask=~sat_mask)
                    divided_statistics['%s_std' % key] = splitter_decorator(get_std)(data, circular=key=='hue', mask=~sat_mask)
                else:
                    divided_statistics['%s_avg' % key] = splitter_decorator(get_avg)(data, circular=key=='hue')
                    divided_statistics['%s_std' % key] = splitter_decorator(get_std)(data, circular=key=='hue')
                    
            # NO. COLORS                    
            divided_statistics['n_unique_colors'] = splitter_decorator(get_unique_colors)(np_rgb)
            divided_statistics['pct_white'] = splitter_decorator(get_pct_color)(np_rgb,(255,255,255))
            divided_statistics['pct_black'] = splitter_decorator(get_pct_color)(np_rgb,(0,0,0))
            divided_statistics['pct_grayscale'] = splitter_decorator(lambda x: np.mean(x.flatten()))(sat_mask)

            # END LOOP
            statistics['divided_statistics'] = divided_statistics

            statistics['source_filename'] = fn
            statistics['target_filename'] = os.path.join(
                options['processed_directory'],
                subreddit,
                '%s.png' % (reddit_id)

            )
            statistics['filesize'] = file_size
            statistics['filetype'] = filetype            
            standardized_image = standardize_image(rgb_frame, options)            
            standardized_image.save(statistics['target_filename'])
                    
            insert_statistics_to_db(statistics, cur, options)
            
        conn.commit()
    except KeyboardInterrupt:
        print('Exiting on keyboardinterrupt')
        exit()
            
    except Exception as e:
        if verbose:
            print('ID: %s' % reddit_id)
            print('SUBREDDIT: %s' % subreddit)
            print('FILENAME: %s' % fn)
            print(e)
            print(traceback.format_exc())
        if verbose == 'ERROR_DETECT':
            raise e
        return None

    return fn

## processing functions for image data

# decorator to split into quadrants/other divisions

def split_by(nx,ny=None, other_vars={}):
    if ny is None:
        ny = nx
    def wrapper(func):
        @wraps(func)
        def splits(data, *args, **kwargs):
            size_x, size_y = data.shape[:2]
            x_increment = size_x // nx
            y_increment = size_y // ny
            sub_frame = {}
            kwargs_copy = deepcopy(kwargs)
            dims = len(data.shape)
            for xi in range(nx):
                for yi in range(ny):
                    xmin = xi * x_increment
                    xmax = min((xi+1)*x_increment, size_x)
                    ymin = yi * y_increment
                    ymax = min((yi+1)*y_increment, size_y)
                    if dims == 2:
                        subdata = data[xmin:xmax,ymin:ymax]
                    elif dims == 3:
                        subdata = data[xmin:xmax,ymin:ymax,:]
                    else:
                        raise NotImplementedError(
                            'Split only available for dims=2 and '
                            'dims=3'
                        )
                    for kwarg in list(kwargs_copy.keys()):
                        if kwarg in other_vars:
                            kval = kwargs_copy[kwarg]
                            ksize_x, ksize_y = kval.shape[:2]
                            kx_increment = ksize_x // nx
                            ky_increment = ksize_y // ny
                            kdims = len(kval.shape)
                            kxmin = xi * kx_increment
                            kxmax = min((xi+1)*kx_increment, ksize_x)
                            kymin = yi * y_increment
                            kymax = min((yi+1)*ky_increment, ksize_y)

                            if kdims == 2:
                                kwargs[kwarg] = kwargs_copy[kwarg][kxmin:kxmax,kymin:kymax]
                            elif kdims == 3:
                                kwargs[kwarg] = kwargs_copy[kwarg][kxmin:kxmax,kymin:kymax,:]
                            else:
                                raise NotImplementedError('Cannot divide data of these dimensions')
                                
                    sub_frame[(xi, yi)] = func(
                        subdata,
                        *args,
                        **kwargs
                    )
            return sub_frame
        return splits
    return wrapper

        

# get average value
def get_avg(data, channel=-1, circular = False, mask = None):
    if circular:
        func = circmean
    else:
        func = np.mean
    if channel != -1:
        x = data[:,:,channel].flatten()
        if mask is not None:
            x = x[mask.flatten()]
        return func(x)
    else:
        x = data.flatten()
        if mask is not None:
            x = x[mask.flatten()]
        return func(x)


# std
def get_std(data, channel=-1, circular = False, mask = None):

    if circular:
        func = circstd
    else:
        func = np.std
        
    if channel != -1:
        x = data[:,:,channel].flatten()
        if mask is not None:
            x = x[mask.flatten()]
        return func(x)
    else:
        x = data.flatten()
        if mask is not None:
            x = x[mask.flatten()]
            
        return func(x)


# get % color (white/black)
def get_pct_color(
        data,
        vals=(0,0,0)
):
    return np.mean(
        (data[:,:,0].flatten() == vals[0]) &
        (data[:,:,1].flatten() == vals[1]) &
        (data[:,:,2].flatten() == vals[2])
    )

# get total # of values
# by assigning each color a unique number
def get_unique_colors(
        data
):
    return len(set((
        data[:,:,0] * 256 +
        data[:,:,1] +
        data[:,:,2] / 256
    ).flatten().tolist()))


def initialize_tables(options, cur = None, conn=None):
    if cur is None or conn is None:
        conn = make_conn()
        cur = conn.cursor()

    tablename = options['data_tablename']

    if '.' in tablename:
        schema, _ = tablename.split('.')
        cur.execute("CREATE SCHEMA IF NOT EXISTS {schema}".format(schema=schema))
        conn.commit()

    division_template = """
    div_{dframe}_n_unique_colors BIGINT,
    div_{dframe}_red_avg FLOAT,
    div_{dframe}_blue_avg FLOAT,
    div_{dframe}_green_avg FLOAT,
    div_{dframe}_hue_avg FLOAT,
    div_{dframe}_saturation_avg FLOAT,
    div_{dframe}_value_avg FLOAT,
    div_{dframe}_red_std FLOAT,
    div_{dframe}_blue_std FLOAT,
    div_{dframe}_green_std FLOAT,
    div_{dframe}_hue_std FLOAT,
    div_{dframe}_saturation_std FLOAT,
    div_{dframe}_value_std FLOAT,
    div_{dframe}_pct_white FLOAT,
    div_{dframe}_pct_black FLOAT,
    div_{dframe}_pct_grayscale FLOAT,
    """

    # NAME FRAMES AS IDX1_IDX2

    division_query_part = '\n'.join([
        division_template.format(
            dframe='%s_%s' % (xi, yi)
        )
        for xi in range(options['nx'])
        for yi in range(options['ny'])        
    ])

    query = """
    CREATE TABLE IF NOT EXISTS {tablename}(
      id SERIAL PRIMARY KEY,
      reddit_id VARCHAR(8),
      subreddit VARCHAR(24),

      source_filename VARCHAR(256),
      target_filename VARCHAR(256),

      frame_idx INTEGER,
      n_frames INTEGER DEFAULT -1,
      height INTEGER,
      width INTEGER,
      filetype VARCHAR(8),

      n_unique_colors BIGINT,
      red_avg FLOAT,
      blue_avg FLOAT,
      green_avg FLOAT,
      hue_avg FLOAT,
      saturation_avg FLOAT,
      value_avg FLOAT,
      red_std FLOAT,
      blue_std FLOAT,
      green_std FLOAT,
      hue_std FLOAT,
      saturation_std FLOAT,
      value_std FLOAT,
      pct_white FLOAT,
      pct_black FLOAT,
      pct_grayscale FLOAT,

      {division_columns}

      -- other metadata to be filled in
      filesize BIGINT,
      score INTEGER
    )
    """.format(
        tablename=tablename,
        division_columns = division_query_part
    )

    cur.execute(query)
    conn.commit()

    return 0
    


def main(options):
    print(options)

    # TODO: filter out images that haven't been processed yet
    fn_list = [
        os.path.join(options['image_directory'], subdir, fn)
        for subdir in os.listdir(options['image_directory'])
        for fn in  os.listdir(os.path.join(options['image_directory'], subdir))
    ]
    hue_dict = dict()

    # get existing ids
    conn = make_conn()
    cur = conn.cursor()
    initialize_tables(options, cur=cur, conn=conn)    

    cur.execute("SELECT DISTINCT reddit_id FROM {tablename}".format(
        tablename=options['data_tablename']
    ))

    existing_ids = set([x[0] for x in cur.fetchall()])
    print('N EXISTING IDS: %d' % len(existing_ids))
    
    fn_list = [
        fn for fn in fn_list
        if re.sub(r'.*/([a-z0-9]+)\..*',r'\1', fn) not in existing_ids
    ]

    print(len(fn_list))
    

    # TEMPORARY
    #fn_list = fn_list[:16]

    subreddit_list = [
        re.sub(r'.*/([A-Za-z_0-9-]+)/[a-z0-9]+\..*',r'\1', fn)
        for fn in fn_list
    ]

    print(fn_list[:8])
    print(subreddit_list[:8])
    
    # ZIP THESE UP
    fn_subreddits = list(zip(fn_list, subreddit_list))
    #exit()

    # TODO: make & call different function
    # the function ojn
    processing_function = partial(process_image, options=options, verbose=True)
    with Pool(options['n_processes']) as p:
        res_maps = p.starmap(processing_function, fn_subreddits)

    print(res_maps[:64])

    print('Done!')


def standardize_image(img, options):
    # create blank image (RGBA = 0)
    out_w, out_h = options['output_width'], options['output_height']
    blank_np = np.zeros((out_h, out_w, 4))
    # resize image to fit into largest dim
    img_w, img_h = img.size
    target_ratio = out_w/out_h
    current_ratio = img_w/img_h
    ratio_ratio = current_ratio/target_ratio
    # process
    if ratio_ratio == 1:
        # just resize image, no borders needed
        new_img = img.resize((out_w, out_h))
    elif ratio_ratio < 1:
        # width to height is smaller on current, so padding needs to be on sides
        adjusted_w = int(img_w * out_h/img_h)
        temp_img = np.asarray(img.resize((adjusted_w, out_h)))
        w_difference = out_w - adjusted_w
        w_start = w_difference // 2
        blank_np[:,w_start:w_start+adjusted_w,:3] = temp_img
        blank_np[:,w_start:w_start+adjusted_w,3] = 255
        new_img = Image.fromarray(np.uint8(blank_np), 'RGBA')
    else:
        # width to height is larger on current, so padding needs to be on top/bottom
        adjusted_h = int(img_h * out_w/img_w)
        temp_img = np.asarray(img.resize((out_w, adjusted_h)))
        h_difference = out_h - adjusted_h
        h_start = h_difference // 2
        blank_np[h_start:h_start+adjusted_h,:,:3] = temp_img
        blank_np[h_start:h_start+adjusted_h,:,3] = 255
        new_img = Image.fromarray(np.uint8(blank_np), 'RGBA')
    # return image for writing in main summarizing/processing function
    return new_img


def get_options():
    parser = argparse.ArgumentParser(
        description='Process and summarize downloaded image data'
    )

    parser.add_argument(
        '--image-directory',
        help='Image directory to process data from'
    )

    parser.add_argument(
        '--processed-directory',
        help='Directory to store processed images in'
    )

    parser.add_argument(
        '--data-tablename',
        help='Name of SQL table to store data in. '
        'Necessary schema will be created if it does not exist.'
    )

    parser.add_argument(
        '--nx',
        type=int,
        default=2,
        help='Number of vertical cells to divide image into '
        'for summarizing statistics'
    )

    parser.add_argument(
        '--ny',
        type=int,
        default=2,
        help='Number of horizontal cells to divide image into '
        'for summarizing statistics'
    )

    parser.add_argument(
        '--output-height',
        default=480,
        type=int,
        help = 'Output image height'
    )

    parser.add_argument(
        '--output-width',
        default=320,
        type=int,
        help='Output image width'
    )

    parser.add_argument(
        '--n-processes',
        default=4,
        type=int,
        help='Number of multithreading processes to use'
    )

    parser.add_argument(
        '--saturation-mask-threshold',
        default=0,
        type=int,
        help='Values less than or equal to this will not be used for hue calculations'
    )

    args = parser.parse_args()
    options = vars(args)

    return options

if __name__=='__main__':
    options = get_options()
    main(options)
