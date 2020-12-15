# How to Run Code

0. Set up PostgreSQL database and change parameters/functionality in `pg_conn.py`

1. Scrape data (see scraping example below)

2. Data was exported from postgres to disk.

3. Run `grab_art.py` with desired parameters

4. Run `process_image_data.py` with desired parameters

5. Under construction for further steps

## Note

xzipped data used in the analysis is included in the repository. Use `xz` to decompress the files.

# Scraping example

Data was scraped using my [TreeGrabForReddit scraper](https://github.com/mcandocia/TreeGrabForReddit) and this command:

    python3 scraper.py pixelart_dec2020 --skip-comments --user-comment-limit 0 --user-thread-limit 0 --type top --limit 1000 -s PixelArt

More subreddits are being used for further analysis.


# Caveats

Only .png, .jpg, .gif, and .bmp images were processed. Video files were not.

