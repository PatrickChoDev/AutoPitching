# import required libraries
import sounddevice as sd
import numpy as np


freq = 16000
duration = 0.5

myarray = np.zeros((int(freq*duration), 2))

while True:
    myrecording = sd.rec( freq, channels=2)
    sd.wait()
    print(myrecording.shape)