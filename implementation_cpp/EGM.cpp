#include "shared_functions.h"

RecordIterates *EGM(const Params &p)
{
	int size_x = (int)p.c.rows();
	int size_y = (int)p.b.rows();
	Iterates iter(size_x, size_y);
	auto record = new RecordIterates(size_x, size_y, p.max_iter / p.record_every);

	while (true)
	{
		EGMStep(iter, p, *record);
		AdaptiveRestarts(iter, p, *record);
		if (iter.count > p.max_iter)
			break;
	}

	// std::cout << iter.z << std::endl;
	return record;
}

void EGMStep(Iterates &iter, const Params &p, RecordIterates &record)
{
	iter.x_hat = (iter.x - p.eta * compute_F(iter.x, iter.y, p).head(iter.size_x)).cwiseMax(0);
	iter.y_hat = iter.y - p.eta * compute_F(iter.x, iter.y, p).tail(iter.size_y);

	iter.x = iter.x - p.eta * compute_F(iter.x_hat, iter.y_hat, p).cwiseMax(0);
	iter.y = iter.y - p.eta * compute_F(iter.x_hat, iter.y_hat, p);

	iter.update(p.restart);

	if ((iter.count - 1) % p.record_every == 0 || (iter.count - 1) % p.print_every == 0)
	{
		iter.compute_convergence_information(p);
		if ((iter.count - 1) % p.record_every == 0)
			record.append(iter, p);
		if ((iter.count - 1) % p.print_every == 0)
			iter.print_iteration_information(p);
	}
}