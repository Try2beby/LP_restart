#include "shared_functions.h"

void process_argument(int argc, char *argv[], Params &p);

int main(int argc, char *argv[])
{
	Params p;
	process_argument(argc, argv, p);

	p.max_iter = 5e5;
	p.print_every = 100;
	p.save2file = 0;
	p.print_timing = false;
	p.set_verbose(1, 0);
	p.load_pagerank();
	if (p.precondition)
	{
		p.scaling();
	}
	SpMat KKT = p.K * p.K.transpose();
	double sigma_max = std::sqrt(PowerIteration(KKT, 1)); // 1 for verbose
	std::cout << sigma_max << std::endl;
	p.eta = 1 / sigma_max;
	// p.eta = 1e-3;
	// p.eta_hat = 1 / (p.K.cwiseAbs() * Eigen::VectorXd::Ones(p.K.cols())).maxCoeff();
	// std::cout << p.eta_hat << std::endl;
	// p.eta_hat = 5e-4;
	auto record = PDHG(p);
	delete record;
	return 0;
}

void process_argument(int argc, char *argv[], Params &p)
{
	std::map<std::string, int> arg_map = {{"adaptive_step_size", 0}, {"restart", 0}, {"primal_weight_update", 0}, {"presolve", 0}, {"scaling", 0}, {"tol", -8}};
	for (int i = 1; i < argc; i += 2)
	{
		arg_map[argv[i]] = atoi(argv[i + 1]);
	}
	p.adaptive_step_size = arg_map["adaptive_step_size"];
	p.restart = arg_map["restart"];
	p.primal_weight_update = arg_map["primal_weight_update"];
	p.precondition = arg_map["scaling"];
	p.eps = std::pow(10, arg_map["tol"]);
}

// void process_argument(int argc, char *argv[], Method &m, Params &p);

// int main(int argc, char *argv[])
// {
// 	if (argc < 2)
// 	{
// 		std::cout << "Usage: ./LP_restart -m method -d dataidx -r restart -l fixed_restart_length" << std::endl;
// 		std::cout << "method: ADMM, PDHG, EGM" << std::endl;
// 		std::cout << "dataidx: 0, 1, 2, 3" << std::endl;
// 		std::cout << "restart: 0, 1" << std::endl;
// 		std::cout << "fixed_restart_length: positive int" << std::endl;
// 		exit(0);
// 	}
// 	if ((argc - 1) % 2)
// 	{
// 		std::cout << "Specify parameters in pairs" << std::endl;
// 	}
// 	// redirect output to file
// 	auto path = projectpath + outputpath + std::string(argv[2]) + "_" +
// 				std::string(argv[4]) + "_" + std::string(argv[6]) + "_" +
// 				std::string(argv[8]) + ".txt";
// 	std::filesystem::create_directories(projectpath + outputpath);
// 	std::ofstream out(path, std::ios::app);
// 	std::cout.rdbuf(out.rdbuf());

// 	Params p;
// 	Method method;
// 	process_argument(argc, argv, method, p);

// 	p.max_iter = 5000;
// 	// p.max_iter = 1e4;
// 	// p.max_iter = 5e5;
// 	p.print_every = 100;
// 	p.save2file = true;
// 	p.print_timing = false;
// 	p.set_verbose(1, 0);
// 	// p.load_model(p.dataidx);
// 	p.load_pagerank();
// 	Eigen::SparseMatrix<double, Eigen::ColMajor> AAT = p.A * p.A.transpose();
// 	double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
// 	p.eta_hat = 1 / sigma_max;
// 	// p.eta = 1e-1;
// 	// p.eta = 10;
// 	// auto w = GetOptimalw(p, PDHG);
// 	p.w = std::pow(4, 2);
// 	auto record = method(p);
// 	delete record;

// 	// close output file
// 	std::cout << std::endl;
// 	std::cout << std::endl;
// 	out.close();

// 	return 0;
// }
