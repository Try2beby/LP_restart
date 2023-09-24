#include "shared_functions.h"

Iterates::Iterates(const int& Size_x, const int& Size_y) : n(0), t(0), count(0)
{
	size_x = Size_x;
	size_y = Size_y;
	size_z = size_x + size_y;
	z = Eigen::VectorXd::Zero(size_z);
	z_hat = Eigen::VectorXd::Zero(size_z);
	z_bar = Eigen::VectorXd::Zero(size_z);
}

void Iterates::update()
{
	z_bar = t * 1.0 / (t + 1) * z_hat + 1.0 / (t + 1) * z_hat;
	t++;
	count++;
}

Eigen::VectorXd Iterates::getx() const
{
	return z.head(size_x);
}

Eigen::VectorXd Iterates::gety() const
{
	return z.tail(size_y);
}

Params::Params() : env(GRBEnv()), eta(1e-2), beta(1e-1), w(1), max_iter(static_cast<int>(5e3)), tau0(1), verbose(false), restart(true)
{
	env.set(GRB_IntParam_OutputFlag, verbose);
}

void Params::set_verbose(const bool& Verbose)
{
	verbose = Verbose;
	env.set(GRB_IntParam_OutputFlag, verbose);
}

void Params::load_model(const std::string& data)
{
	GRBModel model = GRBModel(env, data);
	model.update();

	// Get the number of variables in the model.
	int numVars = model.get(GRB_IntAttr_NumVars);

	// Get the number of constraints in the model.
	int numConstraints = model.get(GRB_IntAttr_NumConstrs);
	// std::cout << numConstraints << std::endl;

	GRBVar* Vars = model.getVars();
	GRBConstr* Constrs = model.getConstrs();

	// Get the object coefficients from the model.
	c = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_Obj, Vars, numVars), numVars);

	// Get the matrix A, use sparse representation.
	Eigen::SparseMatrix<double> A_tmp(numConstraints, numVars);
	std::vector<Eigen::Triplet<double>> triplets;

	// high_resolution_clock::time_point t1 = high_resolution_clock::now();

	for (int i = 0; i < numConstraints; i++)
	{
		for (int j = 0; j < numVars; j++)
		{
			double tmp = model.getCoeff(Constrs[i], Vars[j]);
			if (tmp != 0.0)
			{
				triplets.push_back(Eigen::Triplet<double>(i, j, tmp));
			}
		}
	}

	/*high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "It took" << time_span.count() << " seconds.";*/

	A_tmp.setFromTriplets(triplets.begin(), triplets.end());
	A = A_tmp;
	// std::cout << A.nonZeros()<<std::endl;

	// Get the right-hand side vector from the model.
	b = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints), numConstraints);
}

Eigen::VectorXd QPmodel(const Eigen::VectorXd& linCoeff, const Params& params, const double& eta, const bool& positive)
{
	int size_x = linCoeff.rows();

	GRBModel model = GRBModel(params.env);

	// Create variables
	GRBVar* x;
	if (positive == true)
	{
		x = model.addVars(size_x, GRB_CONTINUOUS);
	}
	else
	{
		x = new GRBVar[size_x];
		for (int i = 0; i < size_x; i++)
		{
			x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
		}
	}

	// Create objective
	GRBQuadExpr objExpr = GRBQuadExpr();
	Eigen::VectorXd x_quaCoeff = (1 / (2 * eta)) * Eigen::VectorXd::Ones(size_x);
	objExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
	objExpr.addTerms(linCoeff.data(), x, size_x);

	// Set objective
	model.setObjective(objExpr, GRB_MINIMIZE);

	model.optimize();
	return Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);
}

double compute_normalized_duality_gap(const Eigen::VectorXd& z0, double& r, const Params& p)
{
	int size_x = p.c.rows();
	int size_y = p.b.rows();

	Eigen::VectorXd x0 = z0.head(size_x);
	Eigen::VectorXd y0 = z0.tail(size_y);

	Eigen::VectorXd y_coeff = p.b - p.A * x0;
	Eigen::VectorXd x_coeff = y0.transpose() * p.A - p.c.transpose();

	double constant = (double)(p.c.transpose() * x0) - (double)(p.b.transpose() * y0);

	// std::cout << y_coeff << x_coeff << constant << std::endl;

	GRBModel model = GRBModel(p.env);

	// Create variables
	GRBVar* x = model.addVars(size_x, GRB_CONTINUOUS);
	GRBVar* y = new GRBVar[size_y];
	for (int i = 0; i < size_y; i++)
	{
		y[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
	}

	// Create objective
	GRBLinExpr objExpr = GRBLinExpr();
	objExpr.addTerms(y_coeff.data(), y, size_y);
	objExpr.addTerms(x_coeff.data(), x, size_x);
	objExpr += constant;

	// Set objective
	model.setObjective(objExpr, GRB_MAXIMIZE);

	// Create constraints
	GRBQuadExpr ConstrExpr = GRBQuadExpr();
	Eigen::VectorXd x_quaCoeff = Eigen::VectorXd::Ones(size_x);
	Eigen::VectorXd y_quaCoeff = Eigen::VectorXd::Ones(size_y);
	Eigen::VectorXd x_linCoeff = -2 * x0;
	Eigen::VectorXd y_linCoeff = -2 * y0;
	ConstrExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
	ConstrExpr.addTerms(y_quaCoeff.data(), y, y, size_y);
	ConstrExpr.addTerms(x_linCoeff.data(), x, size_x);
	ConstrExpr.addTerms(y_linCoeff.data(), y, size_y);
	ConstrExpr += x0.squaredNorm() + y0.squaredNorm();

	// Add constraints
	model.addQConstr(ConstrExpr, GRB_LESS_EQUAL, r * r);

	model.optimize();

	/*std::cout<<model.get(GRB_DoubleAttr_ObjVal)<<std::endl;
	std::cout<<x[0].get(GRB_DoubleAttr_X)<<std::endl;
	std::cout<<y[0].get(GRB_DoubleAttr_X)<<std::endl;*/

	return model.get(GRB_DoubleAttr_ObjVal) / r;
}

double compute_convergence_information(const Iterates& iter, const Params& p)
{
	Eigen::VectorXd x = iter.getx();
	Eigen::VectorXd y = iter.gety();
	Eigen::VectorXd kkt_error_vec;
	kkt_error_vec << -x, p.A* x - p.b, p.b - p.A * x, p.A.transpose()* y - p.c,
		p.c.transpose()* x - p.b.transpose() * y;

	return kkt_error_vec.norm();
}

void AdaptiveRestarts(Iterates& iter, const Params& p, std::vector<Iterates>& IteratesList)
{
	bool restart = false;
	if (iter.n == 0)
	{
		if (iter.t >= p.tau0)
		{
			restart = true;
		}
	}
	else
	{
		// ||z^n,0-z_bar^n,t||
		double r1 = (IteratesList[iter.count - iter.t - 1].z - iter.z_bar).norm();
		int tau_n_minus_1 = IteratesList[iter.count - iter.t - 2].t;
		// ||z^n,0-z^n-1,0||
		double r2 = (IteratesList[iter.count - iter.t - 1].z -
			IteratesList[iter.count - iter.t - tau_n_minus_1 - 2].z).norm();
		double duality_gap1 = compute_normalized_duality_gap(iter.z_bar, r1, p);
		double duality_gap2 = compute_normalized_duality_gap(IteratesList[iter.count - iter.t - 1].z, r2, p);
		if (duality_gap1 <= p.beta * duality_gap2)
		{
			restart = true;
		}
	}

	if (restart == true)
	{
		iter.t = 0;
		iter.n++;
		iter.count++;
		iter.z = iter.z_bar;
		IteratesList[iter.count - 1] = iter;
	}

}

double PowerIteration(const Eigen::SparseMatrix<double>& A, const bool& verbose = false)
{
	int size = A.rows();
	Eigen::VectorXd u = Eigen::VectorXd::Random(size);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(size);
	double tol = 1e-6;
	int max_iter = 1000;
	int iter = 0;
	double lambda = 0;
	double lambda_prev = 0;

	while (true)
	{

		y = u / u.norm();
		u = A * y;
		lambda_prev = lambda;
		lambda = y.transpose() * u;

		iter++;
		if (std::abs(lambda - lambda_prev) / std::abs(lambda) < tol) {
			if (verbose) std::cout << "Power Iteration Converged in " << iter << " iterations." << std::endl;
			break;
		}
		else if (iter >= max_iter) {
			if (verbose) std::cout << "Maximum Iterations Reached." << std::endl;
			break;
		}
	}
	return lambda;
}