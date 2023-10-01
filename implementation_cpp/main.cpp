#include "shared_functions.h"

int main()
{
	using std::cout, std::endl;
	Params p;
	p.set_verbose(1, 0);
	p.max_iter = 5000;
	p.print_every = 10;
	//p.load_example();
	// all data: qap10 qap15 nug08-3rd nug20
	p.load_model("data/qap15.mps");
	Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
	//double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	// p.eta = 0.9 * sigma_max;
	p.eta = 10;
	p.w = std::pow(4, 2);

	p.restart = false;

	ADMM(p);

	return 0;
}