import numpy as np

import sounddevice as sd

import pydub
import pydub.playback

import torchaudio

# First extract the sound from the video file:
# ffmpeg -i summertime-sadness.mov -ab 160k -ac 2 -ar 44100 -vn summertime-sadness.wav
wav_file = '/Users/oddkiva/Desktop/summertime-sadness.wav'
y, sr = torchaudio.load(wav_file)
print(type(y), y.shape, y.dtype, y.device)

wav_audio = pydub.AudioSegment.from_file(wav_file)
# pydub.playback.play(wav_audio)

# y is an array of interleaved left-right channel samples.
y = wav_audio.get_array_of_samples()
sr = wav_audio.frame_rate

# Get the samples from the left channel
y = np.array(y)[::2]
# Normalize
y = np.float32(y) / 10000
y -= np.mean(y)

# Play the left channel at twice the speed
playback_speed = 1
sd.play(y, sr * playback_speed)
# Block until we finished playing the sound
sd.wait()


import ipdb; ipdb.set_trace()
