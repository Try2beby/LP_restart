#include "shared_functions.h"

RecordIterates PDHG(const Params& p)
{
	Iterates iter(p.c.rows(), p.b.rows());
	RecordIterates record(p.c.rows(), p.b.rows(), p.max_iter / p.record_every + 2);
	record.append(iter, p);

	while (true)
	{
		PDHGStep(iter, p, record);
		//AdaptiveRestarts(iter, p, record);
		//FixedFrequencyRestart(iter, p, record, 64);
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
	Eigen::VectorXd x_new = (x - (p.eta / p.w) * (p.c - p.A.transpose() * y)).cwiseMax(0);
	Eigen::VectorXd y_new = y - p.eta * p.w * (-p.b + p.A * (2 * x_new - x));

	iter.z << x_new, y_new;
	iter.z_hat << x_new, y_new;
	iter.update();

	if ((iter.count - 1) % p.record_every == 0 || (iter.count - 1) % p.print_every == 0)
	{
		iter.compute_convergence_information(p);
		if ((iter.count - 1) % p.record_every == 0)
			record.append(iter, p);
		if ((iter.count - 1) % p.print_every == 0)
			iter.print_iteration_information(p);
	}
}
