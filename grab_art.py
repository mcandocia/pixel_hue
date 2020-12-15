import urllib3
import time
import numpy as np
import re
import csv
import os
import argparse
from pg_conn import make_conn

def suffix_from_url(url):
    return re.sub(r'.*\.','', url)

def get_options():
    parser = argparse.ArgumentParser(
        description='Grab data from specific schema in db from '
        'TreeGrabForReddit script as specified in pg_conn.py '
        'settings.'
    )

    parser.add_argument(
        '--schema',
        help='Schema to draw samples from',
        default='multiart_dec2020'
    )

    parser.add_argument(
        '--limit',
        default=1000,
        type=int,
        help='Limit for results'
    )

    parser.add_argument(
        '--suffixes',
        default=['gif','png','jpg','jpeg','bmp'],
        nargs='+',
        help='Valid (lowercased) suffixes for files to retrieve'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.25,
        help='Delay between requests'
    )

    parser.add_argument(
        '--target-directory',
        default='subreddit_images',
        help='Directory path to store images in. Can be relative '
        'or absolute.'
    )


    args = parser.parse_args()

    options = vars(args)

    return options



IMG_DIR = 'images'

# \copy (SELECT id, author_name, url FROM pixelart_dec2020.threads WHERE right(url, 4) IN ('.png','.gif','.jpg','.bmp') AND id IS NOT NULL ORDER BY score DESC LIMIT 600) TO /ntfsl/workspace/pixelart_dl/threads.csv DELIMITER ',' CSV HEADER

def get_existing_ids(options):
    directory = options['target_directory']

    subdirectories = os.listdir(directory)

    existing_ids = [
        re.sub(r'^([a-z0-9]+?)\..*$',r'\1', fn)
        for subdir in subdirectories
        for fn in os.listdir(os.path.join(directory, subdir))
    ]

    return existing_ids


def fetch_data(options):
    assert re.match(r'^\w+$',options['schema'])
    existing_ids = get_existing_ids(options)

    suffix_part = "AND url ~ '.*{suffixes}$'".format(
        suffixes = '(%s)' % '|'.join(options['suffixes'])
    )

    existing_ids_subquery = '(%s)' % ','.join(
        [ascii(x) for x in existing_ids]
    )

    if existing_ids_subquery != '()':
        existing_ids_subquery = (
            'AND TRIM(id) NOT IN ' + existing_ids_subquery
        )
    else:
        existing_ids_subquery = ' '
    
    conn = make_conn()

    cur = conn.cursor()

    query = """
SELECT id, subreddit, url FROM (SELECT id, subreddit, url, is_self, rank() OVER(PARTITION BY subreddit ORDER BY SCORE DESC) AS rank FROM {schema}.threads) A WHERE rank <= {limit} AND id IS NOT NULL AND NOT is_self {suffix_part} {existing_ids_subquery} ORDER BY rank, subreddit
    """.format(
        suffix_part=suffix_part,
        existing_ids_subquery=existing_ids_subquery,
        **options
    )
    assert not ';' in query

    if len(query) < 400:
        print(query)
    else:
        print(query[:200])
        print('...')
        print(query[max(len(query)-200,200):])


    cur.execute(query)

    results = [
        {
            'id': res[0].strip(),
            'subreddit':res[1],
            'url':res[2],
            
        }
        for res in cur.fetchall()
    ]

    return results
    
    
def main(options):
    os.makedirs(options['target_directory'], exist_ok=True)
    
    thread_data = fetch_data(options)
    http = urllib3.PoolManager()
    for i, thread in enumerate(thread_data):
        if (i+1) % 25 == 0:
            print('On iter %s/%s' % (i + 1, len(thread_data)))

        directory = os.path.join(
            options['target_directory'],
            thread['subreddit'],
        )

        os.makedirs(directory, exist_ok=True)
            
        print('%s (%s)' % (thread['id'], thread['subreddit']))
        url = thread['url']
        fn = os.path.join(
            directory,
            '%s.%s' % (thread['id'],suffix_from_url(url))
        )
        if not os.path.isfile(fn):
            status = dl(url, fn, http=http)
            if status:
                print('Could not download. Skipping...')
            else:
                time.sleep(options['delay'])
        else:
            print('skipping...')
            
        
        

def make_fn_from_url(url):
    x = re.sub('.*/','', url)
    return x

def dl(url, outfile=None, max_attempts=3, sf=None, http=None):
    if http is None:
        http = urllib3.PoolManager()
    if outfile is None:
        outfile = make_fn_from_url(url)
    n_attempts=0
    while n_attempts < max_attempts:
        n_attempts += 1
        try:
            response = http.request('GET', url)
            content = response.data
            with open(outfile, 'wb') as f:
                f.write(content)
                f.close()
            return 0
        except KeyboardInterrupt:
            print('Exiting from keyboard...')
            exit()
        except Exception as e:
            print(e)
            if sf is not None:
                sf()
            else:
                sleep_time = 0.25 + min(np.random.lognormal(0.9,0.1),10)
                time.sleep(sleep_time)

    return 1
        
if __name__=='__main__':
    options = get_options()
    main(options)
