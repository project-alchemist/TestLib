#include "AlLib.hpp"

using alchemist::Parameters;

namespace allib {

AlLib::AlLib(MPI_Comm & world) : Library(world)
{
	int world_rank, world_size;
	MPI_Comm_rank(world, &world_rank);
	MPI_Comm_size(world, &world_size);

	bool isDriver = world_rank == 0;
	auto name = (isDriver) ? "AlLib driver" : "AlLib worker " + std::to_string(world_rank);

	log = start_log("library", "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l]    %v");

	log->info("Po1 {}/{}", world_rank, world_size);
}

int AlLib::load()
{


//
//	freopen("myfile.out","w",stdout);
//	freopen("myfile.err","w",stderr);
//	log->info("Poooooooo22222222 {}", world_rank);
//	  printf ("This sentence is redirected to a file.");
//	  std::cout << "What about this" << std::endl;
//	  std::cerr << "and this" << std::endl;
//		log->info("Poooooooo33333333 {}", world_rank);
//	  fclose(stdout);
//	  fclose(stderr);
//		log->info("Poooooooo444444444 {}", world_rank);
//	  stdout = fdopen(1, "w");
////	  stderr = fdopen(2, "w");
//		freopen(2,"w",stderr);


//	auto sink = std::make_shared<spdlog::sinks::simple_file_sink_st>("AlLib.log");
//	auto log = std::make_shared<spdlog::logger>(name, sink);
//	log->flush_on(spdlog::level::info);
//	log->set_level(spdlog::level::info);
//
//
//
//
//			auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
//			        console_sink->set_level(spdlog::level::warn);
//			        console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");
//
//			        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
//			        file_sink->set_level(spdlog::level::trace);
//
//			        spdlog::logger logger("multi_sink", {console_sink, file_sink});
//			        logger.set_level(spdlog::level::debug);


	log->info("AlLib loaded");

	return 0;
}

int AlLib::unload()
{
	log->info("AlLib unloaded");

	return 0;
}

int AlLib::run(string & task_name, Parameters & in, Parameters & out)
{
	// Get the rank and size in the original communicator
	int world_rank, world_size;
	MPI_Comm_rank(world, &world_rank);
	MPI_Comm_size(world, &world_size);

	bool is_driver = world_rank == 0;

	if (task_name.compare("greet") == 0) {

		if (is_driver) log->info("Saying hello from TestLib driver");
		else log->info("Saying hello from TestLib worker {}", world_rank);

	}
	else if (task_name.compare("kmeans") == 0) {

//		uint32_t num_centers    = input.get_int("num_centers");
//		uint32_t max_iterations = input.get_int("max_iterations");		// How many iteration of Lloyd's algorithm to use
//		double epsilon          = input.get_double("epsilon");			// if all the centers change by Euclidean distance less
//																		//     than epsilon, then we stop the iterations
//		std::string init_mode   = input.get_string("init_mode");			// Number of initialization steps to use in kmeans||
//
//		uint32_t init_steps     = input.get_int("init_steps");			// Which initialization method to use to choose
//																		//     initial cluster center guesses
//		uint64_t seed           = input.get_long("seed");					// Random seed used in driver and workers
//
//
//		log->info("init_mode {}:", init_mode);
////		KMeans kmeans = new KMeans(num_centers, max_iterations, epsilon, init_mode, init_steps, seed);
//		KMeans * kmeans = new KMeans(log, world, grid);
////		kmeans.set_log(log);
////		kmeans.set_world(world);
////		kmeans.set_peers(peers);
//		kmeans->set_parameters(num_centers, max_iterations, epsilon, init_mode, init_steps, seed);
//		kmeans->set_data_matrix(input.get_distmatrix("data"));
//		kmeans->run(output);
	}
	else if (task_name.compare("truncated_svd") == 0) {

//		uint64_t rank = m.read_uint64();

//		truncated_svd();

//		SVD * svd = new SVD(log, world, grid);
//
//		svd->set_rank(input.get_int("rank"));
//		svd->set_data_matrix(input.get_distmatrix("data"));
//
//		svd->run(output);
	}

	return 0;
}


}
