#ifndef TESTLIB_HPP
#define TESTLIB_HPP

//#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <poll.h>
#include <thread>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <El.hpp>
#include <eigen3/Eigen/Dense>
#include "arpackpp/arrssym.h"
#include "include/Alchemist.hpp"
//#include "nla/nla.hpp"						// Include all NLA routines
//#include "ml/ml.hpp"							// Include all ML/Data-mining routines

namespace alchemist {

struct TestLib : Library {

	TestLib(MPI_Comm & world);

	~TestLib() { }

	Log_ptr log;

	int world_rank;

	int load();
	int unload();

	int run(string & name, std::vector<Parameter_ptr> & in, std::vector<Parameter_ptr> & out);
};

// Class factories
#ifdef __cplusplus
extern "C" {
#endif

void * create_library(MPI_Comm & world) {
	return reinterpret_cast<void*>(new TestLib(world));
}

void destroy_library(void * p) {
	delete reinterpret_cast<TestLib*>(p);
}

#ifdef __cplusplus
}
#endif
}

#endif // TESTLIB_HPP
