#include "shared_functions.h"

void AdaptivePDHGStep(Iterates &iter, Params &p, RecordIterates &record);

RecordIterates *PDHG(Params &p)
{
	std::cout << "using PDHG: " << std::endl;
	int size_x = (int)p.c.rows();
	int size_y = (int)p.q.rows();
	auto record = new RecordIterates(size_x, size_y, p.max_iter / p.record_every);
	Iterates iter(size_x, size_y);
	while (true)
	{
		if (p.adaptive_step_size)
		{
			AdaptivePDHGStep(iter, p, *record);
		}
		else
		{
			PDHGStep(iter, p, *record);
		}
		if (p.restart)
		{
			if (p.fixed_restart_length == -1)
			{
				AdaptiveRestarts(iter, p, *record);
			}
		}

		if (iter.terminate || iter.count > p.max_iter)
		{
			iter.print_iteration_information(p);
			break;
		}
	}
	std::string file_name{"foo"};
	if (p.restart)
	{
		if (p.fixed_restart_length == -1)
		{
			file_name = "adaptive_restarts";
			// record->saveRestart_idx(__func__, p.data_name, file_name);
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
	// record->saveConvergeinfo(__func__, p.data_name, file_name);

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

	for (int i = 0; i < INFINITY; i++)
	{
		// check size of K, c, q
		x_prime = (iter.x - (eta / p.w) * (p.c - p.K.transpose() * iter.y));
		x_prime = x_prime.cwiseMin(p.ub).cwiseMax(p.lb);
		y_prime = iter.y - (eta * p.w) * (-p.q + p.K * (2 * x_prime - iter.x));
		y_prime = p.sense_vec.select(y_prime.cwiseMax(0), y_prime);

		eta_bar = std::pow(PDHGnorm(iter.x - x_prime, iter.y - y_prime, p.w), 2.0) /
				  std::abs(2 * (y_prime - iter.y).transpose() * p.K * (x_prime - iter.x));
		eta_prime = std::min((1 - std::pow(iter.count + 1, -0.3)) * eta_bar,
							 (1 - std::pow(iter.count + 1, -0.6)) * eta);
		// std::cout << "eta: " << eta << std::endl;
		// std::cout << "eta_bar: " << eta_bar << std::endl;
		// std::cout << std::abs(2 * (y_prime - iter.y).transpose() * p.K * (x_prime - iter.x)) << std::endl;
		// std::cout << (y_prime - iter.y).norm() << " " << (x_prime - iter.x).norm() << std::endl;
		// std::cout << "eta_prime: " << eta_prime << std::endl;
		if (eta <= eta_bar)
		{
			// std::cout << "break" << std::endl;
			iter.x = x_prime;
			iter.y = y_prime;
			p.eta = eta;
			p.eta_hat = eta_prime;
			break;
		}
		eta = eta_prime;
	}

	// std::cout << iter.x.norm() << " " << iter.y.norm() << std::endl;
	iter.update(p);

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

void PDHGStep(Iterates &iter, const Params &p, RecordIterates &record)
{
	Eigen::VectorXd x_new = (iter.x - (p.eta / p.w) * (p.c - p.K.transpose() * iter.y));
	x_new = x_new.cwiseMax(p.lb).cwiseMin(p.ub);
	Eigen::VectorXd y_new = iter.y - (p.eta * p.w) * (-p.q + p.K * (2 * x_new - iter.x));
	// choose idx by sense_vec, if sense_vec(i) == 1, then y(i) = max(0, y(i))
	y_new = p.sense_vec.select(y_new.cwiseMax(0), y_new);

	iter.x = x_new;
	iter.y = y_new;

	std::cout << iter.x.norm() << " " << iter.y.norm() << " " << (p.c - p.K.transpose() * iter.y).norm() << " " << p.c.norm() << std::endl;
	std::cout << iter.x.transpose() << std::endl;

	iter.update(p);

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
