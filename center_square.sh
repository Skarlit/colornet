#!/usr/bin/env bash

FILES=./tmp/raw/*.jpg
ITER=0
PRINT_COUNT=0
for f in $FILES
do
  width=`identify -format "%[fx:min(w,h)]" $f`
  convert $f -gravity Center -crop ${width}x${width}+0+0 -resize 224x224 -fuzz 1% ./tmp/x/${ITER}.jpg
  ITER=$(expr $ITER + 1)
  PRINT_COUNT=$(expr $PRINT_COUNT + 1)
  if [ $PRINT_COUNT -gt 100 ]
  then
     echo $(($ITER * 100))
     PRINT_COUNT=0
  fi
done
