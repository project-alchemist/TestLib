# TestLib

Testing Library for Alchemist

## Dependencies

TestLib requires an implementation of MPI 3.0, for instance MPICH or Open MPI. Don't install more than one implementation.

TestLib also requires the following packages:
* Elemental: For distributing the matrices between Alchemist processes and distributed linear algebra
* spdlog: For thread-safe logging during execution
* Arpack and Arpack-pp: For distributed linear algebra
* Eigen3: For eigenvalue computations


TestLib also requires the following dependencies:

## Installation instructions

### Install prerequisites

If not done so already, mae sure that `gcc`, `make`, `cmake` and implementation of MPI 3.0 are installed.

### Install dependencies

#### Elemental

If already installed, set `ELEMENTAL_PATH` to point to the Elemental directory.

If not installed, set `ELEMENTAL_PATH` to where it should be installed and enter

```
cd $TEMP_DIR && \
git clone git://github.com/elemental/Elemental.git && \
cd Elemental && \
git checkout 0.87 && \
mkdir build && \
cd build && \
CC=gcc-7 CXX=g++-7 FC=gfortran-7 cmake -DCMAKE_BUILD_TYPE=Release (-DEL_IGNORE_OSX_GCC_ALIGNMENT_PROBLEM=ON) -DCMAKE_INSTALL_PREFIX=$ELEMENTAL_PATH .. && \
nice make -j4  && \
make install  && \
cd $TEMP_DIR && \
rm -rf Elemental
```
where the part in parentheses on the 7th line should only be included when installing on a Mac.

#### Install arpack and arpackpp

If already installed, set `ARPACK_PATH` to point to the arpack directory.

If not installed, set `ARPACK_PATH` to where it should be installed and enter

```
cd $TEMP_DIR && \
git clone https://github.com/opencollab/arpack-ng && \
cd arpack-ng && \
git checkout 3.5.0 && \
mkdir build && \
cd build && \
cmake -DMPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$ARPACK_PATH .. && \
nice make -j4 && \
make install && \
cd $TEMP_DIR && \
rm -rf arrack-ng
```
followed by

```
cd $TEMP_DIR && \
git clone https://github.com/m-reuter/arpackpp && \
cd arpackpp && \
git checkout 88085d99c7cd64f71830dde3855d73673a5e872b && \
mkdir build && \
cd build && \
cmake -DCMAKE_INSTALL_PREFIX=${ARPACK_PATH} .. && \
make install && \
cd $TEMP_DIR && \
rm -rf arpackpp
```

#### Install Eigen

If already installed, set `EIGEN3_PATH` to point to the Eigen directory.

If not installed, set `EIGEN3_PATH` to where it should be installed and enter

```
cd $TEMP_DIR && \
curl -L http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 | tar xvfj - && \
cd eigen-eigen-5a0156e40feb && \
mkdir build && \
cd build && \
cmake -DCMAKE_INSTALL_PREFIX=$EIGEN3_PATH .. && \
nice make -j4 && \
make install && \
cd $TEMP_DIR && \
rm -rf eigen-eigen-5a0156e40feb
```

#### Install spdlog

If already installed, set `SPDLOG_PATH` to point to the spdlog directory.

If not installed, set `SPDLOG_PATH` to where it should be installed and enter

```
cd /root && \
git clone https://github.com/gabime/spdlog.git && \
cd spdlog && \
mkdir $SPDLOG_PATH && cp -r include/ $SPDLOG_PATH/include/ && \
cd /root && rm -rf spdlog
```

### Clone the TestLib repo
```
export TESTLIB_PATH=(/desired/path/to/TestLib/directory)
cd $TESTLIB_PATH
git clone https://github.com/project-alchemist/TestLib.git
```

### Update configuration file

In the config.sh file:
* change SYSTEM to the system you are working on;
* set TESTLIB_PATH, ELEMENTAL_PATH, ARPACK_PATH and EIGEN3_PATH to the appropriate paths.

It may also be a good idea to add the above paths to the bash profile.

### Building TestLib

Assuming the above dependencies have been installed and the configuration file updated, TestLib can be built using
```
./build.sh
```

The Alchemist-Client Interfaces (ACIs) will need to know where the TestLib shared library is (`.dylib` on Mac, `.so` on Linux), so it might be a good idea to export a variable that points to it.

## To-Do
1) **Add more functionality**. Currently only truncated SVD. *Expected late September 2019*.
