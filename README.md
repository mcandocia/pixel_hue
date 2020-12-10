# How to Run Code

1. Scrape data (see scraping example below)

2. Data was exported from postgres to disk.

3. Images were retrieved from `grab_art.py`

4. Images were processed with `calculate_hues.py`

5. Data was summarized/visualized with `summarize_hues.r`

## Note

xzipped data used in the analysis is included in the repository. Use `xz` to decompress the files.

# Scraping example

Data was scraped using my [TreeGrabForReddit scraper](https://github.com/mcandocia/TreeGrabForReddit) and this command:

    python3 scraper.py pixelart_dec2020 --skip-comments --user-comment-limit 0 --user-thread-limit 0 --type top --limit 1000 -s PixelArt


# Caveats

Only .png, .jpg, .gif, and .bmp images were processed. Video files were not.

