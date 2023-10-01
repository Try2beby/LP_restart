#include "shared_functions.h"

void ADMM(const Params& p)
{
	Iterates iter(2, p.c.rows(), p.c.rows());
	RecordIterates record(2, p.c.rows(), p.c.rows(), p.max_iter / p.record_every + 2);
	record.append(iter, p);
	Cache cache;
	cache.z_cur_start = iter.z;

	std::vector<GRBModel> model;
	generate_update_model(p, model);

	while (true)
	{
		ADMMStep(iter, p, record, model);
		AdaptiveRestarts(iter, p, record, cache);
		if (iter.count > p.max_iter)
			break;
	}

	/*std::cout << iter.getxU().head(100) << std::endl;
	std::cout << iter.getxV().head(100) << std::endl;*/
}

void ADMMStep(Iterates& iter, const Params& p, RecordIterates& record, std::vector<GRBModel>& model)
{
	int size_x = p.c.rows();

	Eigen::VectorXd xU_prev = iter.getxU();
	Eigen::VectorXd xV_prev = iter.getxV();
	Eigen::VectorXd y_prev = iter.gety();

	Eigen::VectorXd xU = update_x(Eigen::VectorXd::Zero(size_x), 1.0,
		-xV_prev - (1.0 / p.eta) * y_prev, p.eta, model[0], p.verbose,
		iter.count + 1, p.print_every);
	Eigen::VectorXd xV = update_x(p.c, -1.0,
		xU - (1.0 / p.eta) * y_prev, p.eta, model[1], p.verbose, iter.count + 1, p.print_every);
	Eigen::VectorXd y = y_prev - p.eta * (xU - xV);
	iter.z_hat << xU, xV, y_prev - p.eta * (xU - xV_prev);
	iter.z << xU, xV, y;

	iter.update();

	if ((iter.count - 1) % p.record_every == 0)
		record.append(iter, p);
	if ((iter.count - 1) % p.print_every == 0)
	{
		iter.print_iteration_information(p);
	}
}

void generate_update_model(const Params& p, std::vector<GRBModel>& model)
{
	int size_x = p.c.rows();
	// model_xU
	GRBModel model_xU = GRBModel(p.env);
	GRBVar* xU = new GRBVar[size_x];
	for (int i = 0; i < size_x; i++)
	{
		xU[i] = model_xU.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
	}
	for (int i = 0; i < p.A.rows(); i++) {
		GRBLinExpr expr;
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(p.A, i); it; ++it) {
			expr += it.value() * xU[it.col()];
		}
		model_xU.addConstr(expr == p.b[i]);
	}
	model_xU.update();

	GRBModel model_xV = GRBModel(p.env);
	GRBVar* xV = model_xV.addVars(size_x, GRB_CONTINUOUS);
	model_xV.update();

	model.push_back(model_xU);
	model.push_back(model_xV);
}

Eigen::VectorXd update_x(const Eigen::VectorXd& theta, const double& coeff,
	const Eigen::VectorXd& constant, const double& eta, GRBModel& model,
	const bool& verbose, const int& count, const int& print_every)
{
	int size_x = theta.size();
	//get variables
	GRBVar* x = model.getVars();

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
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	if (verbose && (count - 1) % print_every == 0) std::cout << "model.optimize() takes " << duration << " microseconds" << std::endl;

	return Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);
}