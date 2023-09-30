#include "shared_functions.h"

using namespace std::chrono;

void PDHGStep(Iterates&, const Params&, RecordIterates&);
RecordIterates PDHG(const Params&);

//int main()
//{
//	using std::cout, std::endl;
//	Params p;
//	p.set_verbose(true, false);
//	p.max_iter = 5000;
//	p.print_every = 100;
//	p.load_model("data/nug08-3rd.mps");
//	Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
//	double sigma_max = std::sqrt(PowerIteration(AAT, 1)); // 1 for verbose
//	// p.eta = 0.9 * sigma_max;
//	p.eta = 1e-1;
//
//	cout << std::setprecision(9) << sigma_max << endl;
//	p.w = std::pow(4, 2);
//	p.restart = false;
//
//	PDHG(p);
//	// double optimal_w = GetOptimalw(p, PDHG);
//
//	return 0;
//}

RecordIterates PDHG(const Params& p)
{
	Iterates iter(p.c.rows(), p.b.rows());
	RecordIterates record(p.c.rows(), p.b.rows(), p.max_iter / p.record_every + 2);
	record.append(iter, p);
	Cache cache;
	cache.z_cur_start = iter.z;
	while (true)
	{
		PDHGStep(iter, p, record);
		if (iter.count % p.evaluate_every == 0)
			AdaptiveRestarts(iter, p, record, cache);
		if (iter.count > p.max_iter)
			break;
	}
	// std::cout << iter.z << std::endl;
	return record;
}

void PDHGStep(Iterates& iter, const Params& p, RecordIterates& record)
{
	Eigen::VectorXd x = iter.getx();
	Eigen::VectorXd y = iter.gety();
	double eta_x = p.eta / p.w;
	double eta_y = p.eta * p.w;
	Eigen::VectorXd x_new = (x - eta_x * (p.c - p.A.transpose() * y)).cwiseMax(0);
	Eigen::VectorXd y_new = y - eta_y * (-p.b + p.A * (2 * x_new - x));

	iter.z << x_new, y_new;
	iter.z_hat << x_new, y_new;
	iter.update();

	if ((iter.count - 1) % p.record_every == 0)
		record.append(iter, p);
	if ((iter.count - 1) % p.print_every == 0)
	{
		iter.print_iteration_information(p);
	}
}

// void PrimalDualMethods()
//{
// }
