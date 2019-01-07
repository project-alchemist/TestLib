#!/bin/bash

source ./config.sh

if [ "$SYSTEM" == "MacOS" ]
then
	export TESTLIB=$TESTLIB_PATH/target/testlib.dylib"
else
	export TESTLIB=$TESTLIB_PATH/target/testlib.so"
fi
export TESTLIB=$TESTLIB_PATH/target/testlib

CURR_DIR=$PWD

echo " "
cd $TESTLIB_PATH
echo "Building TestLib for $SYSTEM"
LINE="====================="
for i in `seq 1 ${#SYSTEM}`;
do
	LINE="$LINE="
done
echo $LINE
echo " "
echo "Creating TestLib shared object:"
echo " "
cd ./build/$SYSTEM/
make
cd ../..
echo " "
echo $LINE
echo " "
echo "Building process for TestLib has completed"
echo " "
echo "If no issues occurred during build:"
echo "  TestLib shared object located at:    $TESTLIB"
echo " "
cd $CURR_DIR
