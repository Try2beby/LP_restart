#include "shared_functions.h"

typedef RecordIterates *(*method)(const Params &);
std::map<std::string, method> method_map = {{"ADMM", ADMM}, {"PDHG", PDHG}, {"EGM", EGM}};

int main(int argc, char *argv[])
{
	auto method = method_map[argv[1]];
	int dataidx = atoi(argv[2]);
	bool restart = atoi(argv[3]);
	Params p;
	p.restart = restart;
	// p.max_iter = 1e4;
	p.max_iter = 5e5;
	p.print_every = 100;
	p.save2file = false;
	p.print_timing = false;
	p.set_verbose(1, 0);
	p.load_model(dataidx);
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
