#include "shared_functions.h"

std::vector<std::string> Data = { "qap10", "qap15", "nug08-3rd", "nug20" };
std::string cachepath = "cache/";
std::string datapath = "data/";
std::string suffix = ".mps";

void OPT(int dataidx);

int main()
{
	OPT(2);
}

//int main()
//{
//	Eigen::SparseMatrix<double> A(4, 4);
//	A.insert(0, 0) = 1;
//	A.insert(1, 1) = 1;
//	A.insert(2, 2) = 1;
//	A.insert(3, 3) = 1;
//	A.insert(3, 0) = 1;
//	A.makeCompressed();
//
//	Eigen::VectorXd b(4);
//	b << 1, 2, 3, 4;
//
//	Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
//
//	solver.analyzePattern(A);
//	std::cout << solver.info() << std::endl;
//	std::cout << "ok" << std::endl;
//	solver.factorize(A);
//	Eigen::VectorXd x = solver.solve(b);
//	std::cout << x << std::endl;
//
//	return 0;
//}

void OPT(int dataidx)
{
	using std::cout, std::endl;
	Params p;
	p.set_verbose(1, 0);
	p.max_iter = 5000;
	p.print_every = 100;
	p.load_model(datapath + Data[dataidx] + suffix);
	//Eigen::SparseMatrix<double, Eigen::ColMajor> AAT = p.A * p.A.transpose();
	//double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
	//std::cout << sigma_max << std::endl;
	//p.eta = 0.9 * 8;
	p.eta = 1e-1;
	//auto w = GetOptimalw(p, PDHG);
	p.w = std::pow(4, 2);

	//std::vector<int> FixedRestartLengthList = { 64,256,1024 };

	PDHG(p);
}