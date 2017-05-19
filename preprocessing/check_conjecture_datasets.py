def func():
	f1 = open("song_lengths.txt", "r")
	data1 = f1.read()
	data1 = data1.split('\n')
	f1.close()
	f2 = open("real_trans_lengths.txt", "r")
	data2 = f2.read()
	data2 = data2.split('\n')
	f2.close()

	i = 0
	j = 0
	f = open("conjecture_1.txt", "w+")
	while (i < 1000):
		if (data1[i] == data2[j]):
			s1 = int(data1[i+1])
			s2 = int(data1[i+2])
			t1 = int(data2[j+1])
			res1 = s1 + t1/2
			res2 = s2 + t1/2
			res = min(res1, res2)
			f.write(str(res) + "\n")
		else:
			break
		i+=3
		j+=2	
	f.close()
func()
