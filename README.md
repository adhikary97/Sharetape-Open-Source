# ScrapeTape

This Repo is based off of [Sharetape](https://github.com/adhikary97/Sharetape-Open-Source) by Adhikary97

This script uses NLP to determine the 10 best topics from a podcast or long form video. Then based on these topics it cuts 30-60 second clips for TikTok, Instagram, or YouTube Shorts. You can even crop the video to 9:16 format and add captions if needed.

## Demo

[Sharetape Open Source Demo](https://www.youtube.com/watch?v=lPDF0VG9sbk)

## Install homebrew dependencies

```
$ brew install ffmpeg
$ brew install imagemagick
```

## Install requirements

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Download Vosk Library

This is the language library this speech to text uses. Download this [Here](https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip)

Once downloaded unzip in your project directory.

## Execution
There are 4 flags
- `-v` or `-video` is the video path. Only `.mov` or `.mp4`
- `-c` or `-crop` crops the video to 9:16 format, default is `False`
- `-ca` or `-captions` adds captions, default is `False`
- `-cl` or `-clipLength` is length of clip, default is `30 seconds`

Clips will be output in a directory with a unique ID i.e. `c088af43-362b-4837-bf29-d9122008f457/clips`

Example:
```
$ python main.py -v 1086final.mov -c True -ca True -cl 60
```


## FAQ

- if you get the error:
```
Resource stopwords not found. Please use the NLTK Downloader to obtain the resource: >>> import nltk >>> nltk.download('stopwords') For more information see: https://www.nltk.org/data.html Attempted to load corpora/stopwords
```
open a python shell and run:
```
import nltk
nltk.download('stopwords')
```