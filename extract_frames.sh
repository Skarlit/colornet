#!/usr/bin/env bash

ITER=0
FILES=./tmp/videos/*.mp4
for f in $FILES
do
   ffmpeg -i $f -vf fps=1/4 ./tmp/raw/${ITER}%d.jpg
   ITER=$(expr $ITER + 1)
done