#!/bin/bash
# hacked up by xiaolong, xiaolonw@cs.cmu.edu
# THIS IS A DRIVER THAT SHOULD ONLY BE CALLED FROM latedays_starter.sh
# DO REMEMBER TO MOVE THE DIRECTORIES BEFORE USE!
# Feb 6th, 2016

if [ ! -n "$PROCSTRING" ]
    then
    echo "PROCSTRING UNDEFINED, not running anything"
    exit
fi

if [ ! -n "$BASE_DIR" ]
    then
    echo "BASE_DIR UNDEFINED, not running anything"
    exit
fi



tstamp=`date | tr -s ": " "_"`
OUTPUT_FILER=$BASE_DIR/qlogs/torch.$tstamp.${PROCSTRING}.${HOSTNAME}.$$.output
echo ${OUTPUT_FILER}
touch ${OUTPUT_FILER}
cd $BASE_DIR

mkdir -p logs
mkdir -p qlogs

echo $$
nice python grid_search.py ${PROCSTRING} 2>&1 &> ${OUTPUT_FILER}
