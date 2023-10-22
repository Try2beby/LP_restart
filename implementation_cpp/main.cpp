#include "shared_functions.h"

void OPT(RecordIterates &(*method)(const Params &p), int dataidx);

int main()
{
	// for (int i = 0; i <= 2; i++)
	// {
	// 	std::cout << "using data " << Data[i] << std::endl;
	// 	OPT(PDHG, i);
	// }
	OPT(PDHG, 0);
	return 0;
}

void OPT(RecordIterates &(*method)(const Params &p), int dataidx)
{
	using std::cout, std::endl;
	Params p;
	p.max_iter = 100;
	p.print_every = 100;
	p.tol = 1e-5;
	p.set_verbose(0, 0);
	p.load_model(dataidx);
	Eigen::SparseMatrix<double, Eigen::ColMajor> AAT = p.A * p.A.transpose();
	double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	// p.eta = 0.9 / sigma_max;
	p.eta = 1e-1;
	// p.eta = 10;
	// auto w = GetOptimalw(p, PDHG);
	p.w = std::pow(4, 2);
	method(p);

	// std::vector<int> FixedRestartLengthList = { 64,256,1024 };
}