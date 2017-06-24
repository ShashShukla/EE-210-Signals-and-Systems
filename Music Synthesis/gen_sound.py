import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import sounddevice as sd

from helper import *

def generate_independent():
	songfile = "Songs/song_a.txt"
	strip_file(songfile)
	song = get_notes(songfile)
	print song


# =================================================== #
# generate_independent()
# Write code to parse 'song' to make a nx2 array
# that contains the base frequency
# and time when that pluck occured
# Store in pluck as (t,f)

pluck = np.zeros(10)

harmonics = np.array([1., - 1. / 9, 1. / 25])  # relative magnitude of harmonics
adsr = np.array([50, 2000, 0, 0])  # attack,delay,sustain,release
rate = 16  # sampling rate in kHz
total = np.sum(adsr)  # total time for which one pluck persists

def gen_tone(f):
	t = np.arange(total * rate)
	x = np.sin(2 * np.pi * f * t / (1000 * rate))  # Sampling rate = 10kHz
	x = x + harmonics[1] * np.sin(2 * np.pi * 3 * f * t / (1000 * rate))
	x = x + harmonics[2] * np.sin(2 * np.pi * 5 * f * t / (1000 * rate))
	return x


def gen_impulse():
	impulse = np.zeros(total * rate)  # The waveform of the envelope
	for i in range(total * rate):
		if(i < adsr[0] * rate):
			impulse[i] = (1.0 * i / rate) / adsr[0]
		elif(i < (adsr[0] + adsr[1]) * rate):
			impulse[i] = 1.0 * (adsr[0] + adsr[1] - 1.0 * i / rate) / adsr[1]
		# Add code for sustain and release also
	return impulse


pluck = np.zeros(10)

harmonics = np.array([1., - 1. / 9, 1. / 25])  # relative magnitude of harmonics
adsr = np.array([50, 2000, 0, 0])  # attack,delay,sustain,release
rate = 16  # sampling rate in kHz
total = np.sum(adsr)  # total time for which one pluck persists

# print total
impulse = gen_impulse()
plt.plot(impulse)
# plt.show()

tone = gen_tone(180)
plt.plot(tone[:200])
# plt.show()

out = np.multiply(impulse, tone)
plt.plot(out[:30000])
# plt.show()
sd.play(out)