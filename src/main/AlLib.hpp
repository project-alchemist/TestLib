#ifndef ALLIB_HPP
#define ALLIB_HPP

//#include <omp.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <poll.h>
#include <thread>
#include <El.hpp>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "arpackpp/arrssym.h"
//#include <cmath>
//#include <chrono>
//#include <cassert>
//#include <cstdlib>
//#include <cstdio>
//#include <memory>
//#include <unistd.h>
//#include <arpa/inet.h>
#include "include/Alchemist.hpp"
//#include "nla/nla.hpp"						// Include all NLA routines
//#include "ml/ml.hpp"							// Include all ML/Data-mining routines

namespace alchemist {

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct GridObj {

	GridObj(MPI_Comm & _peers) : grid(El::mpi::Comm(_peers)) {}

	El::Grid grid;
};

using std::string;

struct TestLib : Library {

	TestLib(MPI_Comm & world);

	~TestLib() { }

	Log_ptr log;

	int world_rank;

	int load();
	int unload();

	int run(string & name, alchemist::Parameters & input, alchemist::Parameters & output);
};

// Class factories
extern "C" void * create(MPI_Comm & world) {
	return reinterpret_cast<void*>(new TestLib(world));
}

extern "C" void destroy(void * p) {
    delete reinterpret_cast<TestLib*>(p);
}

}

#endif // ALLIB_HPP
