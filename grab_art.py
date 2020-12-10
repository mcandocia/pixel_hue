import urllib3
import time
import numpy as np
import re
import csv
import os

IMG_DIR = 'images'

# \copy (SELECT id, author_name, url FROM pixelart_dec2020.threads WHERE right(url, 4) IN ('.png','.gif','.jpg','.bmp') AND id IS NOT NULL ORDER BY score DESC LIMIT 600) TO /ntfsl/workspace/pixelart_dl/threads.csv DELIMITER ',' CSV HEADER

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    with open('threads.csv', 'r') as f:
        reader = csv.DictReader(f)
        thread_data = list(reader)

    for i, thread in enumerate(thread_data):
        if (i+1) % 25 == 0:
            print('On iter %s' % (i + 1))
            
        print(thread['id'])
        url = thread['url']
        fn = os.path.join(
            IMG_DIR,
            '%s.%s' % (thread['id'],url[-3:])
        )
        if not os.path.isfile(fn):
            status = dl(url, fn)
            if status:
                print('Could not download. Skipping...')
            else:
                time.sleep(0.5)
        else:
            print('skipping...')
            
        
        

def make_fn_from_url(url):
    x = re.sub('.*/','', url)
    return x

def dl(url, outfile=None, max_attempts=3, sf=None):
    if outfile is None:
        outfile = make_fn_from_url(url)
    n_attempts=0
    http = urllib3.PoolManager()
    while n_attempts < max_attempts:
        n_attempts += 1
        try:
            response = http.request('GET', url)
            content = response.data
            with open(outfile, 'wb') as f:
                f.write(content)
                f.close()
            return 0
        except urllib2.URLError as e:
            print(e)
            if sf is not None:
                sf()
            else:
                sleep_time = 0.25 + min(np.random.lognorm(0.9,0.1),10)
                time.sleep(sleep_time)

    return 1
        
if __name__=='__main__':
    main()
