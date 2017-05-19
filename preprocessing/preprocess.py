from pydub import AudioSegment
from scipy import io
import numpy as np
import sys

# secs
trans_lens = [15, 30, 60]
song_len = 100
chunks = 25

def downsample_and_store(s_num):
	global s1, s2, m1
	s1.frame_rate = 44100
	s2.frame_rate = 44100
	m1.frame_rate = 44100

	#save
	s1.export("podcasts/s1_"+ str(s_num) +".mp3", format="mp3")
	s2.export("podcasts/s2_"+ str(s_num) +".mp3", format="mp3")
	m1.export("podcasts/podcast_"+ str(s_num) +".mp3", format="mp3")

def get_real_transition(s_num):
	global s1, s2, m1
        s1d = len(s1)
        s2d = len(s2)
        m1d = len(m1)

        song = m1[s1d:-s2d]
        print(m1d)
        print(s1d)
        print(s2d)
        print(len(song))

        saveToFile1(s_num, s1d, s2d, len(song))
        song.export("podcasts/trans_"+ str(s_num) +".mp3", format="mp3")

def saveToFile1(s_num, s1d, s2d, t1d):
        f = open("song_lengths.txt", "a")
        f.write(str(s_num))
        f.write("\n")
        f.write(str(s1d))
        f.write("\n")
        f.write(str(s2d))
        f.write("\n")
        f.write(str(t1d))
        f.write("\n")
        f.close()

def get_training(s_num):
	global s1, s2, m1

        s1d = len(s1)
        s2d = len(s2)
	m1d = len(m1)
        mid = (s1d + m1d - s2d)/2

        real_trans_len = m1d - s1d - s2d
        saveToFile2(s_num, real_trans_len)

        for t in trans_lens:
		# label y
                trans_start = mid - t*1000/2
                song = m1[trans_start:trans_start+t*1000]
                print(len(song))
                song.export("podcasts/training_data/set_" + str(t) + "/music_data/transition" + ".mp3", format="mp3")

		dhex = song._data
		ddec = map(ord, dhex)
		dnp = np.array(ddec)
		my_dict = {}
		my_dict['y'] = dnp
		io.savemat("podcasts/training_data/set_" + str(t) + "/binary_data/y" + ".mat", my_dict)
		
		#song1
		_s1 = m1[:trans_start]
		_s1d = len(_s1)
		diff = (_s1d - song_len*1000)/3
		song = _s1[:chunks*1000]
		song = song + _s1[chunks*1000 + diff:2*chunks*1000 + diff]
		song = song + _s1[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
		song = song + _s1[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
		print(len(song))
		song.export("podcasts/training_data/set_" + str(t) + "/music_data/s1" + ".mp3", format="mp3")
		
		dhex = song._data
		ddec = map(ord, dhex)
		dnp = np.array(ddec)
		
		#song2
		_s2 = m1[trans_start+t*1000:]
		_s2d = len(_s2)
		diff = (_s2d - song_len*1000)/3
		song = _s2[:chunks*1000]
		song = song + _s2[chunks*1000 + diff:2*chunks*1000 + diff]
		song = song + _s2[2*chunks*1000 + 2*diff:3*chunks*1000 + 2*diff]
		song = song + _s2[3*chunks*1000 + 3*diff:4*chunks*1000 + 3*diff]
		print(len(song))
		song.export("podcasts/training_data/set_" + str(t) + "/music_data/s2" + ".mp3", format="mp3")
		
		dhex = song._data
		ddec = map(ord, dhex)
		dnp = np.append(dnp, np.array(ddec))
		my_dict = {}
		my_dict['x'] = dnp
		io.savemat("podcasts/training_data/set_" + str(t) + "/binary_data/x" + ".mat", my_dict)

def saveToFile2(s_num, l):
        f = open("real_trans_lengths.txt", "a")
        f.write(str(s_num))
        f.write("\n")
        f.write(str(l))
        f.write("\n")
        f.close()

# main
s_num = sys.argv[1]

s1 = AudioSegment.from_mp3("podcasts/song" + str(s_num) + "1.mp3")
s2 = AudioSegment.from_mp3("podcasts/song" + str(s_num) + "2.mp3")
m1 = AudioSegment.from_mp3("podcasts/transition" + str(s_num) + ".mp3")

downsample_and_store(s_num)
get_real_transition(s_num)
get_training(s_num)
