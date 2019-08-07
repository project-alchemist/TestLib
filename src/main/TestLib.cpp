#include "TestLib.hpp"

namespace alchemist {

TestLib::TestLib(MPI_Comm & _world) : Library(_world)
{
	int world_rank, world_size;
	MPI_Comm_rank(world, &world_rank);
	MPI_Comm_size(world, &world_size);

	bool is_driver = world_rank == 0;
	auto name = (is_driver) ? "TestLib driver" : "TestLib worker " + std::to_string(world_rank);

	if (is_driver) {
		log = start_log("TestLib driver", "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l]        %^%v%$", regular, white);
	}
	else {
		char buffer[19];
		sprintf(buffer, "TestLib worker-%03d", (uint8_t) world_rank);

		log = start_log(string(buffer), "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l]    %^%v%$", italic, white);
	}

//	log->info("Po1 {}/{}", world_rank, world_size);
}

int TestLib::load()
{
	log->info("TestLib loaded");

	return 0;
}

int TestLib::unload()
{
	log->info("TestLib unloaded");

	return 0;
}

int TestLib::run(string & task_name, vector<Parameter_ptr> & in, vector<Parameter_ptr> & out)
{
	// Get the rank and size in the original communicator
	int world_rank, world_size;
	MPI_Comm_rank(world, &world_rank);
	MPI_Comm_size(world, &world_size);

	bool is_driver = world_rank == 0;

	if (task_name.compare("greet") == 0) {

		uint8_t in_byte = 0;
		char in_char = ' ';
		int16_t in_short = 0;
		int32_t in_int = 0;
		int64_t in_long = 0;
		float in_float = 0.0;
		double in_double = 0.0;
		string in_string = "";

		for (auto it = in.begin(); it != in.end(); it++) {
			if ((*it)->name == "in_byte")
				in_byte = * reinterpret_cast<uint8_t * >((*it)->p);
			else if ((*it)->name == "in_char")
				in_char = * reinterpret_cast<char * >((*it)->p);
			else if ((*it)->name == "in_short")
				in_short = * reinterpret_cast<int16_t * >((*it)->p);
			else if ((*it)->name == "in_int")
				in_int = * reinterpret_cast<int32_t * >((*it)->p);
			else if ((*it)->name == "in_long")
				in_long = * reinterpret_cast<int64_t * >((*it)->p);
			else if ((*it)->name == "in_float")
				in_float = * reinterpret_cast<float * >((*it)->p);
			else if ((*it)->name == "in_double")
				in_double = * reinterpret_cast<double * >((*it)->p);
			else if ((*it)->name == "in_string")
				in_string = * reinterpret_cast<string * >((*it)->p);
		}

		if (is_driver) log->info("TestLib driver received the following input:");
		else log->info("TestLib worker {} received the following input:", world_rank);
		log->info("    {}", (int) in_byte);
		log->info("    {}", in_char);
		log->info("    {}", in_short);
		log->info("    {}", in_int);
		log->info("    {}", in_long);
		log->info("    {}", in_float);
		log->info("    {}", in_double);
		log->info("    {}", in_string);

		if (is_driver) {
			out.push_back(std::make_shared<Parameter>("out_byte", UINT8, reinterpret_cast<void *>(new uint8_t(in_byte))));
			out.push_back(std::make_shared<Parameter>("out_char", CHAR, reinterpret_cast<void *>(new char(in_char))));
			out.push_back(std::make_shared<Parameter>("out_short", UINT16, reinterpret_cast<void *>(new uint16_t(in_short))));
			out.push_back(std::make_shared<Parameter>("out_int", UINT32, reinterpret_cast<void *>(new uint32_t(in_int))));
			out.push_back(std::make_shared<Parameter>("out_long", UINT64, reinterpret_cast<void *>(new uint64_t(in_long))));
			out.push_back(std::make_shared<Parameter>("out_float", FLOAT, reinterpret_cast<void *>(new float(in_float))));
			out.push_back(std::make_shared<Parameter>("out_double", DOUBLE, reinterpret_cast<void *>(new double(in_double))));
			out.push_back(std::make_shared<Parameter>("out_string", STRING, reinterpret_cast<void *>(new string(in_string))));
		}
		MPI_Barrier(world);
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

		if (is_driver) {

			int rank = 0;
			MatrixInfo * A = nullptr;

			for (auto it = in.begin(); it != in.end(); it++) {
				if ((*it)->name == "rank") {
					rank = (int) * reinterpret_cast<uint32_t * >((*it)->p);
				}
				else if ((*it)->name == "A") {
					A = reinterpret_cast<MatrixInfo * >((*it)->p);
				}
			}


			uint8_t method = 1;

			uint64_t m = A->num_rows;
			uint64_t n = A->num_cols;

			if (rank > m) rank = m;
			if (rank > n) rank = n;

			log->info("Starting truncated SVD on {}x{} matrix", m, n);
			log->info("Settings:");
			log->info("    rank = {}", rank);

			MPI_Barrier(world);

			//	int LOCALEIGS = 0; // TODO: make these an enumeration, and global to Alchemist
			//	int LOCALEIGSPRECOMPUTE = 1;
			//	int DISTEIGS = 2;

			switch(method) {
			case 2:
				log->info("Using distributed matrix-vector products against A, then A tranpose");
				break;
			case 1:
				log->info("Using local matrix-vector products computed on the fly against the local Gramians");
				break;
			case 0:
				log->info("Using local matrix-vector products against the precomputed local Gramians");
				break;
			}

			ARrcSymStdEig<double> prob((int) n, rank, "LM");
			uint8_t command;
			std::vector<double> zerosVector(n);
			for (uint32_t idx = 0; idx < n; idx++)
				zerosVector[idx] = 0.0;

			uint32_t iterNum = 0;

			while (!prob.ArnoldiBasisFound()) {
				prob.TakeStep();
				++iterNum;
				if (iterNum % 20 == 0) log->info("Computed {} matrix-vector products", iterNum);
				if (prob.GetIdo() == 1 || prob.GetIdo() == -1) {
					command = 1;

					MPI_Bcast(&command, 1, MPI_UNSIGNED_CHAR, 0, world);
					if (method == 0 || method == 1) {
						auto temp = prob.GetVector();
						MPI_Bcast(prob.GetVector(), n, MPI_DOUBLE, 0, world);
						MPI_Reduce(zerosVector.data(), prob.PutVector(), n, MPI_DOUBLE, MPI_SUM, 0, world);
						auto temp1 = prob.GetVector();
					}
					if (method == 2) {
		//				MPI_Status status;
		//				MPI_Send(prob.GetVector(), n, MPI_DOUBLE, 1, 0, group);
		//				MPI_Recv(prob.PutVector(), n, MPI_DOUBLE, 1, 0, group, status);
		////				world.send(1, 0, prob.GetVector(), n);
		////				world.recv(1, 0, prob.PutVector(), n);
					}
				}
			}

			prob.FindEigenvectors();
			uint32_t nconv = prob.ConvergedEigenvalues();
			uint32_t niters = prob.GetIter();
			log->info("Done after {} Arnoldi iterations, converged to {} eigenvectors of size {}", niters, nconv, n);

			// NB: it may be the case that n*nconv > 4 GB, then have to be careful!
			// assuming tall and skinny A for now
			Eigen::MatrixXd rightVecs(n, nconv);
			log->info("Allocated matrix for right eigenvectors of A'*A");
			// Eigen uses column-major layout by default!
			for(uint32_t idx = 0; idx < nconv; idx++)
				std::memcpy(rightVecs.col(idx).data(), prob.RawEigenvector(idx), n*sizeof(double));
			log->info("Copied right eigenvectors into allocated storage");

			// Populate U, V, S
			command = 2;
			MPI_Bcast(&command, 1, MPI_UNSIGNED_CHAR, 0, world);
			MPI_Bcast(&nconv, 1, MPI_UNSIGNED, 0, world);
		//	mpi::broadcast(world, nconv, 0);
			log->info("Broadcasted command and number of converged eigenvectors");
			MPI_Bcast(rightVecs.data(), n*nconv, MPI_DOUBLE, 0, world);
		//	mpi::broadcast(world, rightVecs.data(), n*nconv, 0);
			log->info("Broadcasted right eigenvectors");
			auto ng = prob.RawEigenvalues();
			MPI_Bcast(prob.RawEigenvalues(), nconv, MPI_DOUBLE, 0, world);
			log->info("Broadcasted eigenvalues");

			log->info("Waiting on workers to store U, S, and V");

			MPI_Barrier(world);
		}
		else {
			int rank = 0;
			DistMatrix * A = nullptr;

			for (auto it = in.begin(); it != in.end(); it++) {
				if ((*it)->name == "rank") {
					rank = (int) * reinterpret_cast<uint32_t * >((*it)->p);
				}
				else if ((*it)->name == "A") {
					A = reinterpret_cast<DistMatrix * >((*it)->p);
				}
			}

//			for (auto it = in.begin(); it != in.end(); it++) {
//				if ((*it)->name == "rank") {
//					rank = (int) * (* reinterpret_cast<std::shared_ptr<uint32_t> * >((*it)->p));
//				}
//				else if ((*it)->name == "A") {
//					A = * (* reinterpret_cast<std::shared_ptr<DistMatrix_ptr> * >((*it)->p));
//				}
//			}

			uint8_t method = 1;

			const El::Grid & grid = A->Grid();

			int m = A->Height();
			int n = A->Width();

			if (rank > m) rank = m;
			if (rank > n) rank = n;

			MPI_Barrier(world);

			log->info("Starting truncated SVD");

			//	  int LOCALEIGS = 0; // TODO: make these an enumeration, and global to Alchemist
			//	  int LOCALEIGSPRECOMPUTE = 1;
			//	  int DISTEIGS = 2;

			// Assume matrix is row-partitioned b/c relaying it out doubles memory requirements

			//NB: sometimes it makes sense to precompute the gramMat (when it's cheap (we have a lot of cores and enough memory), sometimes
			// it makes more sense to compute A'*(A*x) separately each time (when we don't have enough memory for gramMat, or its too expensive
			// time-wise to precompute GramMat). trade-off depends on k (through the number of Arnoldi iterations we'll end up needing), the
			// amount of memory we have free to store GramMat, and the number of cores we have available
			El::Matrix<double> localGramChunk;

			if (method == 1) {
				localGramChunk.Resize(n, n);
				log->info("Computing the local contribution to A'*A");
				log->info("Local matrix's dimensions are {}x{}", A->LockedMatrix().Height(), A->LockedMatrix().Width());
				log->info("Storing A'*A in {}x{} matrix", n, n);
				auto startFillLocalMat = std::chrono::system_clock::now();
				if (A->LockedMatrix().Height() > 0)
					El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, A->LockedMatrix(), A->LockedMatrix(), 0.0, localGramChunk);
				else
					El::Zeros(localGramChunk, n, n);
				std::chrono::duration<double, std::milli> fillLocalMat_duration(std::chrono::system_clock::now() - startFillLocalMat);
				log->info("Took {} ms to compute local contribution to A'*A", fillLocalMat_duration.count());
			}

			uint8_t command;
			std::unique_ptr<double[]> vecIn{new double[n]};
			El::Matrix<double> localx(n, 1);
			El::Matrix<double> localintermed(A->LocalHeight(), 1);
			El::Matrix<double> localy(n, 1);
			localx.LockedAttach(n, 1, vecIn.get(), 1);
			auto distx = new DistMatrix(n, 1, grid);
			auto distintermed = new DistMatrix(m, 1, grid);
		//	auto distx = El::DistMatrix<double, El::STAR, El::STAR>(n, 1, self->grid);
		//	auto distintermed = El::DistMatrix<double, El::STAR, El::STAR>(m, 1, self->grid);

			log->info("Finished initialization for truncated SVD");

			while(true) {
				MPI_Bcast(&command, 1, MPI_UNSIGNED_CHAR, 0, world);
		//		mpi::broadcast(self->world, command, 0);
				if (command == 1 && method == 0) {
					void * uut;
					MPI_Bcast(vecIn.get(), n, MPI_DOUBLE, 0, world);
					auto uu = vecIn.get();
					El::Gemv(El::NORMAL, 1.0, A->LockedMatrix(), localx, 0.0, localintermed);
					El::Gemv(El::TRANSPOSE, 1.0, A->LockedMatrix(), localintermed, 0.0, localy);
					MPI_Reduce(localy.LockedBuffer(), uut, n, MPI_DOUBLE, MPI_SUM, 0, world);
				}
				if (command == 1 && method == 1) {
					MPI_Bcast(vecIn.get(), n, MPI_DOUBLE, 0, world);
					void * uut;
					El::Gemv(El::NORMAL, 1.0, localGramChunk, localx, 0.0, localy);
					MPI_Reduce(localy.LockedBuffer(), uut, n, MPI_DOUBLE, MPI_SUM, 0, world);
				}
				if (command == 1 && method == 2) {
		//			El::Zeros(distx, n, 1);
		//			log->info("Computing a mat-vec prod against A^TA");
		//			if (self->world.rank() == 1) {
		//				self->world.recv(0, 0, vecIn.get(), n);
		//				distx.Reserve(n);
		//				for(El::Int row=0; row < n; row++)
		//					distx.QueueUpdate(row, 0, vecIn[row]);
		//			}
		//			else {
		//				distx.Reserve(0);
		//			}
		//			distx.ProcessQueues();
		//			log->info("Retrieved x, computing A^TAx");
		//			El::Gemv(El::NORMAL, 1.0, *workingMat, distx, 0.0, distintermed);
		//			log->info("Computed y = A*x");
		//			El::Gemv(El::TRANSPOSE, 1.0, *workingMat, distintermed, 0.0, distx);
		//			log->info("Computed x = A^T*y");
		//			if(self->world.rank() == 1) {
		//				world.send(0, 0, distx.LockedBuffer(), n);
		//			}
				}
				if (command == 2) {
					uint32_t nconv;
					MPI_Bcast(&nconv, 1, MPI_UNSIGNED, 0, world);

					Eigen::MatrixXd rightEigs(n, nconv);
					MPI_Bcast(rightEigs.data(), n*nconv, MPI_DOUBLE, 0, world);
					Eigen::VectorXd singValsSq(nconv);
					MPI_Bcast(singValsSq.data(), nconv, MPI_DOUBLE, 0, world);
					log->info("Received the right eigenvectors and the eigenvalues");

//					DistMatrix_ptr U    = std::make_shared<El::DistMatrix<double, El::VR, El::STAR>>(m, nconv, grid);
//					DistMatrix_ptr S    = std::make_shared<El::DistMatrix<double, El::VR, El::STAR>>(nconv, 1, grid);
//					DistMatrix_ptr Sinv = std::make_shared<El::DistMatrix<double, El::VR, El::STAR>>(nconv, 1, grid);
//					DistMatrix_ptr V    = std::make_shared<El::DistMatrix<double, El::VR, El::STAR>>(n, nconv, grid);


					El::DistMatrix<double, El::VR, El::STAR> * U    = new El::DistMatrix<double, El::VR, El::STAR>(m, nconv, grid);
					El::DistMatrix<double, El::VR, El::STAR> * S    = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, grid);
					El::DistMatrix<double, El::VR, El::STAR> * Sinv = new El::DistMatrix<double, El::VR, El::STAR>(nconv, 1, grid);
					El::DistMatrix<double, El::VR, El::STAR> * V    = new El::DistMatrix<double, El::VR, El::STAR>(n, nconv, grid);

					log->info("Created new matrix objects to hold U, S, and V");

					// populate V
					for (El::Int rowIdx = 0; rowIdx < n; rowIdx++)
						for (El::Int colIdx = 0; colIdx < (El::Int) nconv; colIdx++)
							if (V->IsLocal(rowIdx, colIdx))
								V->SetLocal(V->LocalRow(rowIdx), V->LocalCol(colIdx), rightEigs(rowIdx,colIdx));
					rightEigs.resize(0,0); // clear any memory this temporary variable used (a lot, since it's on every rank)

					// populate S, Sinv
					for (El::Int idx = 0; idx < (El::Int) nconv; idx++) {
						if (S->IsLocal(idx, 0)) {
							S->SetLocal(S->LocalRow(idx), 0, std::sqrt(singValsSq(idx)));
						}
						if (Sinv->IsLocal(idx, 0))
							Sinv->SetLocal(Sinv->LocalRow(idx), 0, 1/std::sqrt(singValsSq(idx)));
					}
					log->info("Stored V and S");

					// form U
					log->info("Computing A*V = U*Sigma");
					log->info("A is {}x{}, V is {}x{}, U will be {}x{}", A->Height(), A->Width(), V->Height(), V->Width(), U->Height(), U->Width());
					//Gemm(1.0, *workingMat, *V, 0.0, *U, self->log);
					El::Gemm(El::NORMAL, El::NORMAL, 1.0, *A, *V, 0.0, *U);
					log->info("Done computing A*V, rescaling to get U");
					// TODO: do a QR instead to ensure stability, but does column pivoting so would require postprocessing S,V to stay consistent
					El::DiagonalScale(El::RIGHT, El::NORMAL, *Sinv, *U);
					log->info("Computed and stored U");
//
//					out.add_distmatrix("S", S);
//					out.add_distmatrix("U", U);
//					out.add_distmatrix("V", V);

					out.push_back(std::make_shared<Parameter>("S", DISTMATRIX_VR_STAR, reinterpret_cast<void *>(S)));
					out.push_back(std::make_shared<Parameter>("U", DISTMATRIX_VR_STAR, reinterpret_cast<void *>(U)));
					out.push_back(std::make_shared<Parameter>("V", DISTMATRIX_VR_STAR, reinterpret_cast<void *>(V)));

					break;
				}
			}
			MPI_Barrier(world);
		}
		log->info("Completed truncated SVD task");
	}

	return 0;
}

}
