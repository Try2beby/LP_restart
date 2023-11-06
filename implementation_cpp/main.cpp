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
	std::map<std::string, std::string> arg_map = {{"adaptive_step_size", "0"},
												  {"restart", "0"},
												  {"primal_weight_update", "0"},
												  {"presolve", "0"},
												  {"scaling", "0"},
												  {"tol", "-8"},
												  {"data_name", "foo"}};
	for (int i = 1; i < argc; i += 2)
	{
		arg_map[argv[i]] = argv[i + 1];
	}
	p.adaptive_step_size = std::stoi(arg_map["adaptive_step_size"]);
	p.restart = std::stoi(arg_map["restart"]);
	p.primal_weight_update = std::stoi(arg_map["primal_weight_update"]);
	p.precondition = std::stoi(arg_map["scaling"]);
	p.eps = std::pow(10, std::stoi(arg_map["tol"]));
	p.data_name = arg_map["data_name"];
}