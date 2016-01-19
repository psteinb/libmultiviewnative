#/bin/bash


if [[ -e $1 ]];then
   NSLICES=`tiffinfo $1 2> /dev/null |tail -n 30|grep -A20 "TIFF Directory"|grep slices|sed -e 's/slices=\([0-9]\+\)/\1/'`
   WIDTH=`tiffinfo $1 2> /dev/null |tail -n 30|grep -A20 "TIFF Directory"|grep Width|sed -e 's/Image Width.* \([0-9]\+\) .*/\1/'`
   HEIGHT=`tiffinfo $1 2> /dev/null |tail -n 30|grep -A20 "TIFF Directory"|grep Width|sed -e 's/.*Image Length.* \([0-9]\+\)/\1/'`
   echo "z y_or_height x_or_width"
   echo $NSLICES $HEIGHT $WIDTH
else
    echo "$1 does not exist"
fi
