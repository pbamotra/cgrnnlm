make -j5
e=$?
if [[ $e -ne 0 ]]
then
    echo "compiling failed. Exiting"
    exit 1
fi
./latedays_starter.sh 0
sleep 1
exit 1
./latedays_starter.sh 1
sleep 1
./latedays_starter.sh 2
sleep 1
./latedays_starter.sh 3
