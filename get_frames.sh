mkdir $2
ffmpeg -i $1 -vf scale=256:256 -r 1/1 $2/%03d.bmp
