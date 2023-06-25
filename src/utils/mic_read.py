import soundcard as sc
import numpy as np

# get a list of all speakers:
speakers = sc.all_speakers()
# get the current default speaker on your system:
default_speaker = sc.default_speaker()
# get a list of all microphones:
mics = sc.all_microphones()
# get the current default microphone on your system:
default_mic = sc.default_microphone()

print(default_speaker)
print(default_mic)


# record and play back one second of audio:
fs = 16000
rec_sec = 0.5

data = default_mic.record(samplerate=fs, numframes=fs*rec_sec)
default_speaker.play(data/np.max(data), samplerate=fs)
print(data)

myarray = np.zeros((int(fs*rec_sec), 2))
# alternatively, get a `Recorder` and `Player` object
# and play or record continuously:
with default_mic.recorder(samplerate=fs) as mic, \
        default_speaker.player(samplerate=fs) as sp:
    while True:
        myrecording = sc.playrec(myarray, fs, channels=2)
        # sp.play(data)
        print(myrecording)

# EOF
