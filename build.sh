#!/bin/bash

source ./config.sh

export ALLIB_SO=$ALLIB_PATH/target/allib

CURR_DIR=$PWD

echo " "
cd $ALLIB_PATH
echo "Building AlLib for $SYSTEM"
LINE="=================="
for i in `seq 1 ${#SYSTEM}`;
do
	LINE="$LINE="
done
echo $LINE
echo " "
echo "Creating AlLib shared object:"
echo " "
cd ./build/$SYSTEM/
make
cd ../..
echo " "
echo $LINE
echo " "
echo "Building process for AlLib has completed"
echo " "
echo "If no issues occurred during build:"
echo "  AlLib shared object located at:    $ALLIB_PATH/target/allib.dylib"
echo " "
cd $CURR_DIR
