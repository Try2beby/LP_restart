#include "shared_functions.h"

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

std::vector<std::string> Data = { "qap10", "qap15", "nug08-3rd", "nug20" };
std::string cachepath = "cache/";
std::string datapath = "data/";
std::string suffix = ".mps";

void OPT(int dataidx);

int main()
{
	OPT(0);
}


void OPT(int dataidx)
{
	using std::cout, std::endl;
	Params p;
	p.set_verbose(1, 0);
	p.max_iter = 5000;
	p.print_every = 10;
	//p.load_example();
	p.load_model(datapath + Data[dataidx] + suffix);
	/*Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	solver.analyzePattern(AAT);
	solver.factorize(AAT);*/
	//double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	// p.eta = 0.9 * sigma_max;
	p.eta = 100;
	p.w = std::pow(4, 2);

	p.restart = false;

	ADMM(p);
}