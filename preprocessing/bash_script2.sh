#!/bin/bash

num_folders="$(aws s3 ls podcastsdatabase/dataset/ | wc -l)"
((num_folders-=2))
echo $num_folders

dfn="podcasts"
ufn="dataset"
d="aws s3 cp s3://podcastsdatabase/dataset/"
u="aws s3 cp $dfn/training_songs s3://podcastsdatabase/dataset/"
u2="aws s3 cp $dfn/training_transitions s3://podcastsdatabase/dataset/"
g="python get_equal_len_training_data.py"
count=1

for i in $(seq 168 $num_folders)
do
	echo $i
	l="$(aws s3 ls podcastsdatabase/dataset/"$i"/ | awk -F" " '{$1=$2=$3=""; print $0}' | cut -c4-)"
	IFS=$'\n'

	link_exists=0
	for item in $l
	do
		link_exists=1
		echo "item: $item"
		if [[ $item == *".metadata"* ]]; then
			continue
		elif [[ $item == "song"*"1.mp3"* ]]; then
			c1="$d$i/$item $dfn/s1.mp3"
		elif [[ $item == "song"*"2.mp3"* ]]; then
			c1="$d$i/$item $dfn/s2.mp3"
		elif [[ $item == "transition$i.mp3" ]]; then
			c1="$d$i/$item $dfn/music.mp3"
		else
			continue
			c1="$d$i/$item $dfn/something.mp3"
		fi
		
		#podcast download
		eval $c1
	
	
		#de-setup
	#	com="rm -r $dfn/*"
	#	eval $com
	#	com="rm -r $ufn/*"
	#	eval $com
	done

	if [[ $link_exists == 0 ]]; then
		continue
	fi

	#python
	c1="$g $i"
	eval $c1
	
	#upload dataset
	c1="$u$i/training_songs/ --recursive"
	eval $c1
	c1="$u2$i/training_transitions/ --recursive"
	eval $c1
	
	rm podcasts/*
	rm podcasts/training_songs/*
	rm podcasts/training_transitions/*
done
