#include "shared_functions.h"

typedef RecordIterates *(*Method)(const Params &);
std::map<std::string, Method> method_map = {{"ADMM", ADMM}, {"PDHG", PDHG}, {"EGM", EGM}};

void process_argument(int argc, char *argv[], Method &m, Params &p);

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: ./LP_restart -m method -d dataidx -r restart -l fixed_restart_length" << std::endl;
		std::cout << "method: ADMM, PDHG, EGM" << std::endl;
		std::cout << "dataidx: 0, 1, 2, 3" << std::endl;
		std::cout << "restart: 0, 1" << std::endl;
		std::cout << "fixed_restart_length: positive int" << std::endl;
		exit(0);
	}
	if ((argc - 1) % 2)
	{
		std::cout << "Specify parameters in pairs" << std::endl;
	}
	Params p;
	Method method;
	process_argument(argc, argv, method, p);

	// p.max_iter = 1e4;
	p.max_iter = 5e5;
	p.print_every = 100;
	p.save2file = true;
	p.print_timing = false;
	p.set_verbose(1, 0);
	p.load_model(p.dataidx);
	Eigen::SparseMatrix<double, Eigen::ColMajor> AAT = p.A * p.A.transpose();
	double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	p.eta = 0.9 / sigma_max;
	// p.eta = 1e-1;
	// p.eta = 10;
	// auto w = GetOptimalw(p, PDHG);
	p.w = std::pow(4, 2);
	auto record = method(p);
	delete record;
	// std::vector<int> FixedRestartLengthList = { 64,256,1024 };
	return 0;
}

void process_argument(int argc, char *argv[], Method &m, Params &p)
{
	std::map<std::string, int> arg_map = {{"-d", 0}, {"-r", 1}, {"-l", -1}};
	for (int i = 3; i < argc; i += 2)
	{
		arg_map[argv[i]] = atoi(argv[i + 1]);
	}
	m = method_map[argv[2]];
	p.dataidx = arg_map["-d"];
	p.restart = arg_map["-r"];
	p.fixed_restart_length = arg_map["-l"];
}
