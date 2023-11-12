#include "shared_functions.h"

RecordIterates *ADMM(Params &p)
{
	std::cout << "using ADMM: " << std::endl;
	std::cout << "eta: " << p.eta << std::endl;
	auto size_x = (int)p.c.rows();
	auto size_y = size_x;
	Iterates iter(2, size_x, size_y);
	auto record = new RecordIterates(2, size_x, size_y, p.max_iter / p.record_every);

	std::vector<GRBModel> model;
	generate_update_model(p, model);
	Eigen::SparseMatrix<double> KKT = p.K * p.K.transpose();
	if (p.verbose)
	{
		// print K rows and cols
		std::cout << "K rows: " << p.K.rows() << " K cols: " << p.K.cols() << std::endl;
		// print nnz of KKT
		std::cout << "KKT number of nonzeros: " << KKT.nonZeros() << std::endl;
	}
	Solver solver;

	Timer timer;
	solver.compute(KKT);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "factorize failed" << std::endl;
		exit(0);
	}
	std::cout << "factorize done, takes " << timer.timing() << " milliseconds" << std::endl;
	// std::cout << "#iterations:     " << solver.iterations() << std::endl;
	// std::cout << "estimated error: " << solver.error() << std::endl;

	while (true)
	{
		// ADMMStep(iter, p, *record, model);
		ADMMStep(iter, p, *record, solver);
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
	record->saveConvergeinfo(__func__, p.data_name, file_name);

	return record;
}

void ADMMStep(Iterates &iter, const Params &p, RecordIterates &record,
			  Solver &solver)
{
	Eigen::VectorXd xU_prev = iter.xU;
	Eigen::VectorXd xV_prev = iter.xV;
	Eigen::VectorXd y_prev = iter.y;

	iter.xU = p.K.transpose() * solver.solve(p.q +
											 p.K * (-xV_prev - (1.0 / p.eta) * y_prev)) -
			  (-xV_prev - (1.0 / p.eta) * y_prev);
	// print xU.norm()
	// std::cout << "xU norm: " << iter.xU.norm() << std::endl;

	iter.xV = ((iter.xU - (1.0 / p.eta) * y_prev) - (1.0 / p.eta) * p.c).cwiseMax(0);
	iter.y = y_prev - p.eta * (iter.xU - iter.xV);
	iter.xU_hat = iter.xU;
	iter.xV_hat = iter.xV;
	iter.y_hat = y_prev - p.eta * (iter.xU - xV_prev);

	iter.update(p);

	auto count = iter.count;
	if ((count - 1) % p.record_every == 0 || (count - 1) % p.print_every == 0)
	{
		iter.compute_convergence_information(p);
		if ((count - 1) % p.record_every == 0)
			record.append(iter, p);
		if ((count - 1) % p.print_every == 0 && p.verbose)
			iter.print_iteration_information(p);
	}
}

void ADMMStep(Iterates &iter, const Params &p, RecordIterates &record,
			  std::vector<GRBModel> &model)
{
	int size_x = (int)p.c.rows();

	Eigen::VectorXd xU_prev = iter.xU;
	Eigen::VectorXd xV_prev = iter.xV;
	Eigen::VectorXd y_prev = iter.y;

	iter.xU = update_x(Eigen::VectorXd::Zero(iter.size_x), 1.0,
					   -xV_prev - (1.0 / p.eta) * y_prev, p.eta, model[0], p.verbose,
					   iter.count + 1, p.print_every);
	iter.xV = update_x(p.c, -1.0,
					   iter.xU - (1.0 / p.eta) * y_prev, p.eta, model[1], p.verbose, iter.count + 1, p.print_every);
	iter.y = y_prev - p.eta * (iter.xU - iter.xV);
	iter.xU_hat = iter.xU;
	iter.xV_hat = iter.xV;
	iter.y_hat = y_prev - p.eta * (iter.xU - xV_prev);

	iter.update(p);

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

void generate_update_model(const Params &p, std::vector<GRBModel> &model)
{
	int size_x = (int)p.c.rows();
	// model_xU
	GRBModel model_xU = GRBModel(p.env);
	GRBVar *xU = new GRBVar[size_x];
	for (int i = 0; i < size_x; i++)
	{
		xU[i] = model_xU.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
	}
	for (int i = 0; i < p.A.rows(); i++)
	{
		GRBLinExpr expr;
		// for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(p.A, i); it; ++it)
		// {
		// 	expr += it.value() * xU[it.col()];
		// }
		model_xU.addConstr(expr == p.b[i]);
	}
	model_xU.update();

	GRBModel model_xV = GRBModel(p.env);
	GRBVar *xV = model_xV.addVars(size_x, GRB_CONTINUOUS);
	model_xV.update();

	model.push_back(model_xU);
	model.push_back(model_xV);
}

Eigen::VectorXd update_x(const Eigen::VectorXd &theta, const double &coeff,
						 const Eigen::VectorXd &constant, const double &eta, GRBModel &model,
						 const bool &verbose, const int &count, const int &print_every)
{
	int size_x = (int)theta.size();
	// get variables
	GRBVar *x = model.getVars();

	// Set objective
	GRBQuadExpr objExpr = GRBQuadExpr();
	Eigen::VectorXd x_quaCoeff = (eta / 2) * Eigen::VectorXd::Ones(size_x);
	Eigen::VectorXd x_linCoeff = eta * coeff * constant + theta;
	objExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
	objExpr.addTerms(x_linCoeff.data(), x, size_x);
	model.setObjective(objExpr, GRB_MINIMIZE);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	model.optimize();
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	if (verbose && (count - 1) % print_every == 0)
		std::cout << "model.optimize() takes " << duration << " milliseconds" << std::endl;

	return Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);
}