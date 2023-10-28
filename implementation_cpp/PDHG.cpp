#include "shared_functions.h"

RecordIterates *PDHG(const Params &p)
{
	int size_x = (int)p.c.rows();
	int size_y = (int)p.b.rows();
	Iterates iter(size_x, size_y);
	auto record = new RecordIterates(size_x, size_y, p.max_iter / p.record_every);
	while (false)
	{
		PDHGStep(iter, p, *record);
		if (p.restart)
		{
			if (p.fixed_restart_length == -1)
			{
				AdaptiveRestarts(iter, p, *record);
			}
			else
			{
				FixedFrequencyRestart(iter, p, *record);
			}
		}

		if (iter.terminate || iter.count > p.max_iter)
			break;
	}
	std::string file_name{"foo"};
	if (p.restart)
	{
		if (p.fixed_restart_length == -1)
		{
			file_name = "adaptive_restarts";
			record->saveRestart_idx(__func__, p.dataidx, file_name);
		}
		else
		{
			file_name = "fixed_restarts_" + std::to_string(p.fixed_restart_length);
		}
	}
	else
	{
		file_name = "no_restarts";
	}
	record->saveConvergeinfo(__func__, p.dataidx, file_name);

	return record;
}

double PDHGnorm(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const int w)
{
	return std::sqrt(w * x.squaredNorm() + (1.0 / w) * y.squaredNorm());
}

void AdaptivePDHGStep(Iterates &iter, Params &p, RecordIterates &record)
{
	Eigen::VectorXd x_prime, y_prime;
	double eta = p.eta_hat, eta_bar, eta_prime;
	while (true)
	{
		x_prime = (iter.x - (eta / p.w) * (p.c - p.A.transpose() * iter.y)).cwiseMax(0);
		y_prime = iter.y - (eta * p.w) * (-p.b + p.A * (2 * x_prime - iter.x));
		eta_bar = std::pow(PDHGnorm(iter.x - x_prime, iter.y - y_prime, p.w), 2.0);
		eta_prime = std::min((1 - std::pow(iter.count + 1, 0.3)) * eta_bar,
							 (1 + std::pow(iter.count + 1, -0.6)) * eta);
		if (eta <= eta_bar)
		{
			iter.x = x_prime;
			iter.y = y_prime;
			p.eta = eta;
			p.eta_hat = eta_prime;
			break;
		}
		eta = eta_prime;
	}
}

void PDHGStep(Iterates &iter, const Params &p, RecordIterates &record)
{
	Timer timer;
	Eigen::VectorXd x_new = (iter.x - (p.eta / p.w) * (p.c - p.A.transpose() * iter.y)).cwiseMax(0);
	Eigen::VectorXd y_new = iter.y - p.eta * p.w * (-p.b + p.A * (2 * x_new - iter.x));
	timer.timing();

	iter.x = x_new;
	iter.y = y_new;

	iter.x_hat = x_new;
	iter.y_hat = y_new;
	iter.update(p.restart);
	timer.timing();

	timer.save(__func__, p, iter.count);

	auto count = iter.count;
	if ((count - 1) % p.record_every == 0 || (count - 1) % p.print_every == 0)
	{
		iter.compute_convergence_information(p);
		if ((count - 1) % p.record_every == 0)
		{
			record.append(iter, p);
		}
		if ((count - 1) % p.print_every == 0 && p.verbose)
		{
			iter.print_iteration_information(p);
		}
	}
}
