from pydub import AudioSegment

def func():
	for j in range(30):
		print(str(j+1))
		i = j+1
		s1 = AudioSegment.from_mp3(str(i) + "/training_songs/s1.mp3")
		s2 = AudioSegment.from_mp3(str(i) + "/training_songs/s2.mp3")
		t1 = AudioSegment.from_mp3(str(i) + "/training_transitions/15.mp3")
		t2 = AudioSegment.from_mp3(str(i) + "/training_transitions/30.mp3")
		t3 = AudioSegment.from_mp3(str(i) + "/training_transitions/60.mp3")
	
		s1.frame_rate = 44100
		s2.frame_rate = 44100
		t1.frame_rate = 44100
		t2.frame_rate = 44100
		t3.frame_rate = 44100
	
		s1.export(str(i) + "/training_songs/s1.mp3", format="mp3")
		s2.export(str(i) + "/training_songs/s2.mp3", format="mp3")
		t1.export(str(i) + "/training_transitions/15.mp3", format="mp3")
		t2.export(str(i) + "/training_transitions/30.mp3", format="mp3")
		t3.export(str(i) + "/training_transitions/60.mp3", format="mp3")

func()
