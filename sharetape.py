import datetime
import json
import logging
import os
import wave

import moviepy.editor as mp
import nltk
import scipy.io.wavfile as wav
import srt
from moviepy.video.tools.subtitles import SubtitlesClip
from nltk.corpus import stopwords
from vosk import KaldiRecognizer

from videocrop import *

TAGS = ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


class Sharetape:
    def __init__(
        self,
        video,
        audio,
        mono_audio,
        transcript,
        words,
        subtitles,
        clip_length,
        crop,
        captions,
        model,
    ) -> None:
        self.video = video
        self.audio = audio
        self.mono_audio = mono_audio
        self.transcript = transcript
        self.words = words
        self.subtitles = subtitles
        self.clip_length = clip_length
        self.crop = crop
        self.captions = captions
        self.model = model

    def load_data(self):
        try:
            with open(self.words, "r") as json_file:
                words = json.load(json_file)
        except:
            words = []
        return words

    def save_data(self, data):
        with open(self.words, "w") as json_file:
            json.dump(data, json_file)

    def cut_video_clip(self, output_file, start_time=0):
        video = mp.VideoFileClip(self.video)

        clip = video.subclip(start_time, min(start_time + self.clip_length, video.end))

        clip.write_videofile(
            output_file, fps=30, threads=5, codec="libx264", verbose=False, logger=None
        )

    def get_topics(self):
        with open(self.transcript, "r") as f:
            transcript = f.read()

        sentences = nltk.sent_tokenize(transcript)

        # Initialize a set of stop words
        stop_words = set(stopwords.words("english"))

        # Initialize a dictionary to store the keyword scores
        keyword_scores = {}

        # Iterate over the sentences and extract the relevant keywords
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [
                word.lower()
                for word in words
                if word.isalpha() and word.lower() not in stop_words
            ]

            for word in words:
                if word in keyword_scores:
                    keyword_scores[word] += 1
                else:
                    keyword_scores[word] = 1

        # Initialize a list to store the relevant topics
        topics = []

        # Iterate over the sentences and calculate the topic scores
        for i, sentence in enumerate(sentences):
            words = nltk.word_tokenize(sentence)
            words = [
                word.lower()
                for word in words
                if word.isalpha() and word.lower() not in stop_words
            ]

            score = 0

            for word in words:
                if word in keyword_scores:
                    score += keyword_scores[word]

            if score > 0:
                topics.append((i, sentence, score))

        # Sort the topics based on score in descending order
        topics = sorted(topics, key=lambda x: x[2], reverse=True)

        return topics

    def remove_overlapping_topics(self, topics):
        retained_topics = []

        for i, (index, sentence, score) in enumerate(topics):
            overlap = False

            for j, (retained_index, retained_sentence, retained_score) in enumerate(
                retained_topics
            ):
                if (
                    len(set(sentence.split()) & set(retained_sentence.split()))
                    / len(set(sentence.split()))
                    > 0.1
                ):
                    overlap = True
                    if score > retained_score:
                        retained_topics[j] = (index, sentence, score)
                    break

            if not overlap:
                retained_topics.append((index, sentence, score))

        return retained_topics

    def match_topic_time(self, topics, words):
        topic_match = []
        for topic in topics:
            topic_list = (topic[1].lower().replace(".", "")).split(" ")
            for ind, word in enumerate(words):
                max_len = len(topic_list)
                if word["word"] == topic_list[0]:
                    count = 0
                    flag = True
                    while count < max_len:
                        if words[ind + count]["word"] != topic_list[count]:
                            flag = False
                            break
                        else:
                            count += 1
                    if flag:
                        topic_dict = {
                            "ind": topic[0],
                            "score": topic[2],
                            "text": topic[1],
                            "word": word,
                        }
                        topic_match.append(topic_dict)

        return topic_match

    def create_clips(self, dir_name, clip_number):
        topics = self.get_topics()
        topics = self.remove_overlapping_topics(topics)

        words_data = self.load_data()

        topic_match = self.match_topic_time(topics, words_data)

        # sort scores in descending order
        topic_match.sort(key=lambda x: x["score"], reverse=True)
        # print(topic_match)

        for t in topic_match[0 : min(len(topic_match), clip_number)]:
            if not self.captions and not self.crop:
                self.cut_video_clip(
                    f'{dir_name}/clips/clip_{dir_name}_{t["ind"]}.mov',
                    t["word"]["start"],
                )
            else:
                self.cut_video_clip_with_captions(
                    f'{dir_name}/clips/clip_{dir_name}_{t["ind"]}.mov',
                    t["word"]["start"],
                )

    def cut_video_clip_with_captions(
        self,
        output_file,
        start_time=0,
        font_style="Verdana",
        font_size=32,
        font_color="white",
        font_stroke_color="white",
        font_stroke_width=1,
        font_box_loc=(800, 110),
        bg_box_loc=(815, 125),
        bg_color=(0, 0, 0),
        bg_opacity=1,
        final_vid_y_pos=900,
    ):
        video = mp.VideoFileClip(self.video)
        end = min(start_time + self.clip_length, video.end)
        clip = video.subclip(start_time, end)

        post_path = ""
        clip_vertical = clip
        if self.crop:
            post_path = (
                f"{output_file.split('.')[0]}_pre_vertical.{output_file.split('.')[1]}"
            )

            clip_vertical = process_video(clip, post_path)

        words = self.load_data()
        filtered = []
        for i in words:
            if end > i["start"] >= start_time:
                filtered.append(i)

        WORDS_PER_LINE = 7
        subs = []
        for j in range(0, len(filtered), WORDS_PER_LINE):
            line = filtered[j : j + WORDS_PER_LINE]
            s = srt.Subtitle(
                index=len(subs),
                content=" ".join([l["word"] for l in line]),
                start=datetime.timedelta(seconds=line[0]["start"] - start_time),
                end=datetime.timedelta(seconds=line[-1]["end"] - start_time),
            )
            subs.append(s)

        subtitle = srt.compose(subs)

        with open(
            f"{output_file.split('.')[0]}_caption.srt", "w+", encoding="utf8"
        ) as f:
            f.writelines(subtitle)

        if not self.crop and self.captions:
            result = self.subtitle_clip(
                clip,
                output_file,
                font_style,
                font_size,
                font_color,
                font_stroke_color,
                font_stroke_width,
                font_box_loc,
                bg_box_loc,
                bg_color,
                bg_opacity,
                final_vid_y_pos,
            )
            result.write_videofile(
                output_file,
                fps=30,
                threads=5,
                codec="libx264",
                verbose=False,
                logger=None,
            )
        elif self.crop and self.captions:
            result_vertical = self.subtitle_clip(
                clip_vertical,
                output_file,
                font_style,
                font_size,
                font_color,
                font_stroke_color,
                font_stroke_width,
                font_box_loc=(400, 110),
                bg_box_loc=(415, 125),
                bg_color=(0, 0, 0),
                bg_opacity=1,
                final_vid_y_pos=900,
            )
            result_vertical.write_videofile(
                f"{output_file.split('.')[0]}_vertical_captions.{output_file.split('.')[1]}",
                fps=30,
                threads=5,
                codec="libx264",
                verbose=False,
                logger=None,
            )
            os.remove(post_path)
        elif self.crop and not self.captions:
            clip_vertical.write_videofile(
                f"{output_file.split('.')[0]}_vertical.{output_file.split('.')[1]}",
                fps=30,
                threads=5,
                codec="libx264",
                verbose=False,
                logger=None,
            )
            os.remove(post_path)

    def subtitle_clip(
        self,
        clip,
        output_file,
        font_style="Verdana",
        font_size=32,
        font_color="white",
        font_stroke_color="white",
        font_stroke_width=1,
        font_box_loc=(800, 110),
        bg_box_loc=(815, 125),
        bg_color=(0, 0, 0),
        bg_opacity=1,
        final_vid_y_pos=900,
    ):
        generator = lambda txt: mp.TextClip(
            txt,
            font=font_style,
            fontsize=font_size,
            color=font_color,
            method="caption",
            stroke_color=font_stroke_color,
            stroke_width=font_stroke_width,
            size=font_box_loc,
        ).on_color(
            size=bg_box_loc, color=bg_color, pos="center", col_opacity=bg_opacity
        )
        subtitles = SubtitlesClip(f"{output_file.split('.')[0]}_caption.srt", generator)
        return mp.CompositeVideoClip(
            [clip, subtitles.set_position(("center", final_vid_y_pos))]
        )

    def extract_transcript(self):
        # extract audio from video. keep commented to use existing audio file
        my_clip = mp.VideoFileClip(self.video)
        if my_clip.audio:
            my_clip.audio.write_audiofile(self.audio, verbose=False, logger=None)

        # transcribe audio file
        transcript, words, _ = self.handle_speech_2_text()

        with open(self.transcript, "w+") as fil:
            fil.write(transcript)

        # save words to file
        self.save_data(words)

    def handle_speech_2_text(self):
        sample_rate, stereo_data = wav.read(self.audio)

        # Extract left and right channels
        left_channel = stereo_data[:, 0]
        right_channel = stereo_data[:, 1]

        # Compute average of left and right channels
        mono_data = (left_channel + right_channel) / 2

        # Convert to integer type
        mono_data = mono_data.astype("int16")

        # Save mono WAV file
        wav.write(self.mono_audio, sample_rate, mono_data)

        wf = wave.open(self.mono_audio, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            logging.error("Audio file must be WAV format mono PCM.")
            return "", "", ""

        rec = KaldiRecognizer(self.model, wf.getframerate())

        rec.SetWords(True)
        rec.SetPartialWords(True)

        transcript = []  # Store the transcript as a list of strings

        results = []
        subs = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())
        results.append(rec.FinalResult())

        WORDS_PER_LINE = 14
        total = []
        total_words = []
        for res in results:
            jres = json.loads(res)
            if not "result" in jres:
                continue
            words = jres["result"]
            total_words.extend(words)
            for j in range(0, len(words), WORDS_PER_LINE):
                line = words[j : j + WORDS_PER_LINE]
                s = srt.Subtitle(
                    index=len(subs),
                    content=" ".join([l["word"] for l in line]),
                    start=datetime.timedelta(seconds=line[0]["start"]),
                    end=datetime.timedelta(seconds=line[-1]["end"]),
                )
                total.append(s.content)
                subs.append(s)

        transcript = ". ".join(total)
        subtitle = srt.compose(subs)

        return (transcript, total_words, subtitle)
