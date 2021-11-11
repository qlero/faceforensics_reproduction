#!/bin/bash

# Command to:
# - convert the network output from .avi to .mp4
# - cut the output at the 4th second
# - add a text box with a description to the output
# - crossfade the source video with the output

ffmpeg -i notMorgan.avi -c:v copy -c:a copy -y notMorgan.mp4
ffmpeg -ss 00:00:04 -i notMorgan.mp4 -to 00:10:00 -c copy nt1.mp4
ffmpeg -i nt1.mp4 -vf "drawtext=fontfile=/path/to/font.ttf:text='Xception c40, on 480p video':fontcolor=white:fontsize=16:box=1:boxcolor=black@0.5:boxborderw=5:x=0.05*w:y=0.05*h" -codec:a copy nt2.mp4
ffmpeg -i ../input_videos/notMorgan.mp4 -i nt2.mp4 -filter_complex \
	"[0]settb=AVTB[v0];[1]settb=AVTB[v1];[v0] \
	[v1]xfade=duration=4:offset=4,format=yuv420p" \
	notMorganCrossfade.mp4
rm -f notMorgan.mp4
rm -f nt1.mp4 nt2.mp4
