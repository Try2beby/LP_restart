#include "shared_functions.h"

void EGMStep(Iterates&, const Params&, RecordIterates&);
void EGM(const Params&);

//int main()
//{
//	using std::cout, std::endl;
//	Params p;
//	p.set_verbose(true, false);
//	p.max_iter = 5000;
//	p.print_every = 100;
//	p.load_model("data/nug08-3rd.mps");
//	Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
//	double sigma_max = std::sqrt(PowerIteration(AAT, 1));  // 1 for verbose
//	//p.eta = 0.9 * sigma_max;
//	p.eta = 1e-1;
//
//	p.restart = false;
//
//	EGM(p);
//
//	return 0;
//}

void EGM(const Params& p)
{
	Iterates iter(p.c.rows(), p.b.rows());
	RecordIterates record(p.c.rows(), p.b.rows(), p.max_iter / p.record_every + 2);
	record.append(iter, p);
	Cache cache;
	cache.z_cur_start = iter.z;
	while (true) {
		EGMStep(iter, p, record);
		AdaptiveRestarts(iter, p, record, cache);
		if (iter.count > p.max_iter) break;
	}

	//std::cout << iter.z << std::endl;
}

void EGMStep(Iterates& iter, const Params& p, RecordIterates& record)
{
	iter.z_hat = iter.z - p.eta * compute_F(iter.z, p);
	iter.z_hat.segment(0, p.c.rows()) = iter.z_hat.segment(0, p.c.rows()).cwiseMax(0);
	iter.z = iter.z - p.eta * compute_F(iter.z_hat, p);
	iter.z.segment(0, p.c.rows()) = iter.z.segment(0, p.c.rows()).cwiseMax(0);

	iter.update();

	if ((iter.count - 1) % p.record_every == 0) record.append(iter, p);
	if ((iter.count - 1) % p.print_every == 0) {
		iter.print_iteration_information(p);
	}
}