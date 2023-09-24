#include <vector>
#include <iostream>
#include <chrono>

using namespace std::chrono;

#include "shared_functions.h"

void PrimalDualStep(Iterates&, const Params&, std::vector<Iterates>&);
double GetOptimalw(Params&);

void PDHG(const Params& p)
{
	Iterates iter(p.c.rows(), p.b.rows());
	std::vector<Iterates> IteratesList(p.max_iter, iter);
	IteratesList[0] = iter; iter.count++;
	while (true) {
		PrimalDualStep(iter, p, IteratesList);
		if (p.restart) AdaptiveRestarts(iter, p, IteratesList);
		if (iter.count >= p.max_iter) break;
	}
}

int main()
{
	using std::cout, std::endl;
	Params p;
	p.set_verbose(true);
	p.load_model("data/qap10.mps");
	Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
	double sigma_max = std::sqrt(PowerIteration(AAT, 1));  // 1 for verbose
	p.eta = 0.9 * sigma_max;

	GetOptimalw(p);

	// test load_model
	/*load_model(env, p);
	cout << (p.A).nonZeros() << endl;*/

	// test QPmodel
	/*p.eta = 0.5;
	Eigen::Matrix<double, 2, 1> p1{2,-4};
	GRBModel model=QPmodel(p1,p,0);
	cout << model.get(GRB_DoubleAttr_ObjVal) << endl;*/

	// test compute_normalized_duality_gap
	/*Eigen::Vector2d z0(0, 0);
	Eigen::Vector2d z1(1, 0);
	p.b = Eigen::VectorXd::Ones(1);
	p.c = -1 * Eigen::VectorXd::Ones(1);
	Eigen::SparseMatrix<double> A(1, 1);
	A.insert(0, 0) = 1;
	p.A = A;
	compute_normalized_duality_gap(z0, z1, p);*/

	return 0;
}

void PrimalDualStep(Iterates& iter, const Params& p, std::vector<Iterates>& IteratesList)
{
	if (p.verbose) std::cout << "iter: " << iter.count << std::endl;

	Eigen::VectorXd x = iter.getx();
	Eigen::VectorXd y = iter.gety();
	Eigen::VectorXd x_linCoeff = p.c - p.A.transpose() * y - (1.0 / p.eta) * x;
	Eigen::VectorXd x_new = (p.eta / p.w * x_linCoeff).cwiseMax(0);
	Eigen::VectorXd y_linCoeff = -p.b + p.A * (2 * x_new - x) - (1.0 / p.eta) * y;
	Eigen::VectorXd y_new = -p.eta * p.w * y_linCoeff;

	iter.z << x_new, y_new;
	iter.z_hat << x_new, y_new;
	iter.update();
	IteratesList[iter.count - 1] = iter;
}


// void PrimalDualMethods()
//{
// }

double GetOptimalw(Params& p)
{
	p.restart = false;
	for (int i = 0; i < 1; i++)
	{
		p.w = std::pow(4, -5 + i);
		PDHG(p);
	}
	return 0;
}