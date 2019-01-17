#!/bin/bash

# Configuration file for building TestLib

# export SYSTEM="MacOS"                # Options: MacOS, Cori, Linux
if [ "$SYSTEM" = "" ]; then
  export SYSTEM="MacOS"
fi

if [ "$SYSTEM" == "MacOS" ]; then
	export TESTLIB_PATH=$HOME/Projects/AlLib
	
	export ELEMENTAL_PATH=$HOME/Software/Elemental
	export EIGEN3_PATH=$HOME/Software/Eigen3
	export ARPACK_PATH=$HOME/Software/ARPACK
	
elif [ "$SYSTEM" == "Cori" ]; then
	export TESTLIB_PATH=$SCRATCH/Projects/TestLib
	
	export ELEMENTAL_PATH=$SCRATCH/Software/Elemental
	export SPDLOG_PATH=$SCRATCH/Software/SPDLog
	export EIGEN3_PATH=$SCRATCH/Software/Eigen3
	export ARPACK_PATH=$SCRATCH/Software/ARPACK
	
elif [ "$SYSTEM" == "Linux" ]; then
	export TESTLIB_PATH=/usr/local/TestLib
	
	export ELEMENTAL_PATH=/usr/local/elemental
	export SPDLOG_PATH=/usr/local/spdlog
	export EIGEN3_PATH=/usr/local/eigen3
	export ARPACK_PATH=/usr/local/arpack	
fi
