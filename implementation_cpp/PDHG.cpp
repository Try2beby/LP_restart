#include "shared_functions.h"

RecordIterates& PDHG(const Params& p)
{
	Iterates iter(p.c.rows(), p.b.rows());
	static RecordIterates record(p.c.rows(), p.b.rows(), p.max_iter / p.record_every);

	while (true)
	{
		PDHGStep(iter, p, record);
		AdaptiveRestarts(iter, p, record);
		//FixedFrequencyRestart(iter, p, record, 16384);
		if (iter.terminate || iter.count > p.max_iter)
			break;
	}

	record.saveConvergeinfo(__func__, p.dataidx, "adaptive_restarts");
	record.saveRestart_idx(__func__, p.dataidx, "adaptive_restarts");
	return record;
}

void PDHGStep(Iterates& iter, const Params& p, RecordIterates& record)
{
	auto eta = p.eta; auto w = p.w;
	Eigen::VectorXd x = iter.getx();
	Eigen::VectorXd y = iter.gety();
	Eigen::VectorXd x_new = (x - (eta / w) * (p.c - p.A.transpose() * y)).cwiseMax(0);
	Eigen::VectorXd y_new = y - eta * w * (-p.b + p.A * (2 * x_new - x));

	iter.z << x_new, y_new;
	iter.z_hat << x_new, y_new;
	iter.update();

	auto count = iter.count;
	if ((count - 1) % p.record_every == 0 || (count - 1) % p.print_every == 0)
	{
		iter.compute_convergence_information(p);
		if ((count - 1) % p.record_every == 0)
			record.append(iter, p);
		if ((count - 1) % p.print_every == 0)
			iter.print_iteration_information(p);
	}
}
