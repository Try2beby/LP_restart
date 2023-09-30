#include "shared_functions.h"

void ADMM(const Params&);
void ADMMStep(Iterates&, const Params&, RecordIterates&);
Eigen::VectorXd update_x(const Params&, const Eigen::VectorXd&, const double&,
	const Eigen::VectorXd&, const double&, const bool&);

int main()
{
	using std::cout, std::endl;
	Params p;
	p.set_verbose(true, false);
	p.max_iter = 5000;
	p.print_every = 10;
	p.load_model("data/qap10.mps");
	Eigen::SparseMatrix<double> AAT = p.A * p.A.transpose();
	double sigma_max = std::sqrt(PowerIteration(AAT, 1));  // 1 for verbose
	//p.eta = 0.9 * sigma_max;
	p.eta = 1e-3;

	p.restart = false;

	ADMM(p);

	return 0;
}

void ADMM(const Params& p)
{
	Iterates iter(2, p.c.rows(), p.c.rows());
	RecordIterates record(2, p.c.rows(), p.c.rows(), p.max_iter / p.record_every + 2);
	record.append(iter, p);
	Cache cache;
	cache.z_cur_start = iter.z;
	while (true) {
		ADMMStep(iter, p, record);
		AdaptiveRestarts(iter, p, record, cache);
		if (iter.count > p.max_iter) break;
	}

	//std::cout << iter.z << std::endl;
}

void ADMMStep(Iterates& iter, const Params& p, RecordIterates& record)
{
	int size_x = p.c.rows();

	Eigen::VectorXd xU_prev = iter.getxU();
	Eigen::VectorXd xV_prev = iter.getxV();
	Eigen::VectorXd y_prev = iter.gety();

	Eigen::VectorXd xU = update_x(p, Eigen::VectorXd::Zero(size_x), 1.0,
		-xV_prev - (1.0 / p.eta) * y_prev, p.eta, false);
	Eigen::VectorXd xV = update_x(p, p.c, -1.0,
		xU - (1.0 / p.eta) * y_prev, p.eta, true);
	Eigen::VectorXd y = y_prev - p.eta * (xU - xV);
	iter.z_hat << xU, xV, y_prev - p.eta * (xU - xV_prev);
	iter.z << xU, xV, y;

	iter.update();

	if ((iter.count - 1) % p.record_every == 0) record.append(iter, p);
	if ((iter.count - 1) % p.print_every == 0) {
		iter.print_iteration_information(p);
	}
}

Eigen::VectorXd update_x(const Params& p, const Eigen::VectorXd& theta, const double& coeff,
	const Eigen::VectorXd& constant, const double& eta, const bool& positive)
{
	int size_x = theta.size();
	GRBModel model = GRBModel(p.env);

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

	// Set objective
	GRBQuadExpr objExpr = GRBQuadExpr();
	Eigen::VectorXd x_quaCoeff = (1.0 / (2 * eta)) * Eigen::VectorXd::Ones(size_x);
	Eigen::VectorXd x_linCoeff = eta * coeff * constant - theta;
	objExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
	objExpr.addTerms(x_linCoeff.data(), x, size_x);
	model.setObjective(objExpr, GRB_MINIMIZE);

	// Add constraints
	if (positive == false)
	{
		for (int i = 0; i < p.A.rows(); i++) {
			GRBLinExpr expr = 0;
			for (Eigen::SparseMatrix<double>::InnerIterator it(p.A, i); it; ++it) {
				expr += it.value() * x[it.col()];
			}
			model.addConstr(expr == p.b(i));
		}
	}

	model.optimize();

	Eigen::VectorXd x_new = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);

	return x_new;
}