#include "shared_functions.h"

void OPT(RecordIterates& (*method)(const Params& p), int dataidx);

int main()
{
	OPT(PDHG, 1);
}


void OPT(RecordIterates& (*method)(const Params& p), int dataidx)
{
	using std::cout, std::endl;
	Params p;
	p.max_iter = 50000;
	p.set_verbose(1, 0);
	p.load_model(dataidx);
	//Eigen::SparseMatrix<double, Eigen::ColMajor> AAT = p.A * p.A.transpose();
	//double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	//std::cout << sigma_max << std::endl;
	//p.eta = 0.9 * 8;
	p.eta = 1e-1;
	//auto w = GetOptimalw(p, PDHG);
	p.w = std::pow(4, 2);
	method(p);

	//std::vector<int> FixedRestartLengthList = { 64,256,1024 };
}