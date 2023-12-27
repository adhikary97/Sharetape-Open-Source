import argparse
import os
import uuid

import nltk
from vosk import Model, SetLogLevel

from sharetape import Sharetape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=True, default="")
    parser.add_argument("-vb", "--video_b", type=str, required=False, default="")
    parser.add_argument("-c", "--crop", type=str, required=False, default=False)
    parser.add_argument("-ca", "--captions", type=str, required=False, default=False)
    parser.add_argument("-cl", "--clipLength", type=str, required=False, default=30)
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    SetLogLevel(-1)
    model = Model(model_path="vosk-model-en-us-0.42-gigaspeech")

    video_id = str(uuid.uuid4())
    os.makedirs(f"{video_id}/clips")

    shartape = Sharetape(
        args.video,
        args.video_b,
        f"{video_id}/audio.wav",
        f"{video_id}/mono_audio.wav",
        f"{video_id}/transcript.txt",
        f"{video_id}/words.json",
        f"{video_id}/subtitles.srt",
        int(args.clipLength),
        args.crop == "True",
        args.captions == "True",
        model,
    )
    shartape.extract_transcript()
    shartape.create_clips(video_id, 10)


if __name__ == "__main__":
    main()