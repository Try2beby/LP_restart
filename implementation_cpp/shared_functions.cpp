#include "shared_functions.h"

Timer::Timer()
{
	start_time = high_resolution_clock::now();
}

float Timer::timing()
{
	auto now = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(now - start_time).count();
	start_time = now;
	time_record.push_back(duration);
	return duration;
}

void Timer::save(const std::string method, const Params &p, const int count)
{
	if (p.save2file == true)
	{
		auto path = projectpath + logpath + method + "/" + Data[p.dataidx] + "/";
		std::filesystem::create_directories(path);
		std::ofstream ofs(path + "/time_record.txt", std::ios::app);
		if (ofs.is_open())
		{
			for (auto i : time_record)
			{
				ofs << i << " ";
			}
			ofs << std::endl;
			ofs.close();
		}
		else
			std::cout << "Unable to open file" << std::endl;
	}
	if (p.verbose == true && p.print_timing == true && (count - 1) % p.print_every == 0)
	{
		std::cout << "Timing: ";
		for (auto i : time_record)
		{
			std::cout << i << " ";
		}
		std::cout << "microseconds" << std::endl;
		std::cout << std::endl;
	}
}

Iterates::Iterates(const int &Size_x, const int &Size_y) : n(0), t(0), count(1),
														   terminate{false}
{
	size_x = Size_x;
	size_y = Size_y;
	size_z = size_x + size_y;
	x = Eigen::VectorXd::Zero(size_x);
	y = Eigen::VectorXd::Zero(size_y);
	x_hat = Eigen::VectorXd::Zero(size_x);
	y_hat = Eigen::VectorXd::Zero(size_y);
	x_bar = Eigen::VectorXd::Zero(size_x);
	y_bar = Eigen::VectorXd::Zero(size_y);
	this->use_ADMM = false;
	this->cache.x_cur_start = this->x;
	this->cache.y_cur_start = this->y;
	this->now_time();
}

Iterates::Iterates(const int &Repeat_x, const int &Size_x, const int &Size_y) : n(0), t(0), count(1), terminate{false}
{
	size_x = Size_x;
	size_y = Size_y;
	size_z = Repeat_x * size_x + size_y;
	xU = Eigen::VectorXd::Zero(size_x);
	xV = Eigen::VectorXd::Zero(size_x);
	xU_hat = Eigen::VectorXd::Zero(size_x);
	xV_hat = Eigen::VectorXd::Zero(size_x);
	xU_bar = Eigen::VectorXd::Zero(size_x);
	xV_bar = Eigen::VectorXd::Zero(size_x);
	y = Eigen::VectorXd::Zero(size_y);
	y_hat = Eigen::VectorXd::Zero(size_y);
	y_bar = Eigen::VectorXd::Zero(size_y);
	this->use_ADMM = true;
	this->cache.xU_cur_start = this->xU;
	this->cache.xV_cur_start = this->xV;
	this->cache.y_cur_start = this->y;
	this->now_time();
}

void Iterates::now_time()
{
	time = high_resolution_clock::now();
	start_time = time;
}

float Iterates::timing()
{
	auto now = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(now - time).count();
	time = now;
	return duration / 1e3;
}

float Iterates::end()
{
	auto now = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(now - start_time).count();
	return duration / 1e3;
}

void Iterates::update(const bool restart)
{
	if (use_ADMM == false)
	{
		x_bar = t * 1.0 / (t + 1) * x_bar + 1.0 / (t + 1) * x_hat;
		if (not restart)
		{
			cache.x_prev_start = cache.x_cur_start;
			cache.x_cur_start = this->x;
		}
	}
	else
	{
		xU_bar = t * 1.0 / (t + 1) * xU_bar + 1.0 / (t + 1) * xU_hat;
		xV_bar = t * 1.0 / (t + 1) * xV_bar + 1.0 / (t + 1) * xV_hat;
		if (not restart)
		{
			cache.xU_prev_start = cache.xU_cur_start;
			cache.xU_cur_start = this->xU;
			cache.xV_prev_start = cache.xV_cur_start;
			cache.xV_cur_start = this->xV;
		}
	}
	y_bar = t * 1.0 / (t + 1) * y_bar + 1.0 / (t + 1) * y_hat;
	if (not restart)
	{
		cache.y_prev_start = cache.y_cur_start;
		cache.y_cur_start = this->y;
	}
	t++;
	count++;
}

void Iterates::restart(const Params &p)
{
	n++;
	t = 0;
	count++;
	if (use_ADMM == false)
	{
		x = x_bar;
		cache.x_prev_start = cache.x_cur_start;
		cache.x_cur_start = this->x;
	}
	else
	{
		xU = xU_bar;
		xV = xV_bar;
		cache.xU_prev_start = cache.xU_cur_start;
		cache.xU_cur_start = this->xU;
		cache.xV_prev_start = cache.xV_cur_start;
		cache.xV_cur_start = this->xV;
	}
	y = y_bar;
	cache.y_prev_start = cache.y_cur_start;
	cache.y_cur_start = this->y;

	double r2 = std::sqrt((cache.x_prev_start - cache.x_cur_start).squaredNorm() + (cache.y_prev_start - cache.y_cur_start).squaredNorm());
	cache.gap_cur_prev_start = compute_normalized_duality_gap(cache.x_cur_start, cache.y_cur_start, r2, p);
	compute_normalized_duality_gap(cache.x_cur_start, cache.y_cur_start, r2, p, 0);
}

Convergeinfo Iterates::compute_convergence_information(const Params &p)
{
	Eigen::VectorXd c = p.c;
	Eigen::VectorXd b = p.b;
	Eigen::SparseMatrix<double> A = p.A;

	if (use_ADMM == false)
	{
		Eigen::VectorXd kkt_error_vec(2 * size_x + 2 * size_y + 1);
		kkt_error_vec << -x, A * x - b, b - A * x, A.transpose() * y - c,
			c.dot(x) - b.dot(y);
		this->convergeinfo.kkt_error = (kkt_error_vec.cwiseMax(0)).lpNorm<1>();

		double r{1};
		if (p.restart)
		{
			double r = std::sqrt((x_bar - cache.x_cur_start).squaredNorm() + (y_bar - cache.y_cur_start).squaredNorm());
			this->convergeinfo.normalized_duality_gap = compute_normalized_duality_gap(this->x_bar, this->y_bar, r, p);
			compute_normalized_duality_gap(this->x_bar, this->y_bar, r, p, 0);
		}
		else
		{
			double r = std::sqrt((x - cache.x_prev_start).squaredNorm() + (y - cache.y_prev_start).squaredNorm());
			this->convergeinfo.normalized_duality_gap = compute_normalized_duality_gap(this->x, this->y, r, p);
			// std::cout << this->count - 1 << " r = " << r << std::endl;
		}
	}
	else
	{
		Eigen::VectorXd kkt_error_vec(3 * size_x + 2 * p.b.rows());
		kkt_error_vec << -xV, A * xU - b, b - A * xU, xU - xV, xV - xU;
		this->convergeinfo.kkt_error = (kkt_error_vec.cwiseMax(0)).lpNorm<1>();
	}

	// if (std::abs(this->convergeinfo.normalized_duality_gap) < p.tol || this->convergeinfo.kkt_error < p.tol)
	if (std::abs(this->convergeinfo.normalized_duality_gap) < p.tol)
	{
		this->terminate = true;
		auto duration = this->end();
		std::cout << "Iteration terminates at " << this->count - 1 << ", takes " << duration << "s" << std::endl;
	}
	return this->convergeinfo;
}

void Iterates::print_iteration_information(const Params &p)
{
	std::cout << "Iteration " << count - 1 << ", ";
	std::cout << "kkt_error: " << this->convergeinfo.kkt_error << std::endl;
	std::cout << "normalized_duality_gap: " << this->convergeinfo.normalized_duality_gap << std::endl;

	if (use_ADMM == false)
	{
		std::cout << "obj: " << p.c.dot(x) + p.b.dot(y) - y.transpose() * p.A * x << std::endl;
	}
	else
	{
		std::cout << xU.norm() << " " << xV.norm() << " " << (xU - xV).norm() << " " << y.norm() << std::endl;
		std::cout << "obj: " << p.c.dot(xV) - y.dot(xU - xV) << std::endl;
	}

	std::cout << "Iterations take " << this->timing() << "s" << std::endl;
	std::cout << std::endl;
}

RecordIterates::RecordIterates(const int &Size_x, const int &Size_y, const int &Size_record)
	: end_idx(0)
{
	Iterates iter(Size_x, Size_y);
	std::vector<Iterates> aIteratesList(Size_record, iter);
	std::vector<Convergeinfo> aConvergeinfoList(Size_record);
	IteratesList = aIteratesList;
	ConvergeinfoList = aConvergeinfoList;
	this->use_ADMM = false;
}

RecordIterates::RecordIterates(const int &Repeat_x, const int &Size_x, const int &Size_y, const int &Size_record)
	: end_idx(0)
{
	Iterates iter(Repeat_x, Size_x, Size_y);
	std::vector<Iterates> aIteratesList(Size_record, iter);
	std::vector<Convergeinfo> aConvergeinfoList(Size_record);
	IteratesList = aIteratesList;
	ConvergeinfoList = aConvergeinfoList;
	this->use_ADMM = true;
}

void RecordIterates::append(const Iterates &iter, const Params &p)
{
	IteratesList[end_idx] = iter;
	ConvergeinfoList[end_idx] = iter.convergeinfo;
	end_idx++;
}

Iterates RecordIterates::operator[](const int &i)
{
	if (i < 0 || i >= end_idx)
	{
		throw std::out_of_range("Index out of range");
	}
	return IteratesList[i];
}

void RecordIterates::saveConvergeinfo(const std::string method, const int dataidx, const std::string filename)
{
	auto path = projectpath + cachepath + method + "/" + Data[dataidx] + "/";
	std::filesystem::create_directories(path);
	std::ofstream ofs(path + filename + cachesuffix);
	if (ofs.is_open())
	{
		/*boost::archive::xml_oarchive xml_output_archive(ofs);
		xml_output_archive& BOOST_SERIALIZATION_NVP(this->ConvergeinfoList);*/
		for (int i = 0; i < end_idx; i++)
		{
			auto &obj = ConvergeinfoList[i];
			ofs << std::setprecision(10) << obj.normalized_duality_gap << "," << obj.kkt_error << std::endl;
		}
		std::cout << "save Convergeinfo done" << std::endl;
		ofs.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void RecordIterates::saveRestart_idx(const std::string method, const int dataidx, const std::string filename)
{
	auto path = projectpath + cachepath + method + "/" + Data[dataidx] + "/";
	std::filesystem::create_directories(path);
	std::ofstream ofs(path + filename + "_restart_idx" + cachesuffix);
	if (ofs.is_open())
	{
		for (auto i : restart_idx)
		{
			ofs << i << std::endl;
		}
		std::cout << "save restart_idx done" << std::endl;
		ofs.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

Params::Params() : env(GRBEnv()), eta(1e-1), beta(std::exp(-1)), w(1), tol(1e-7),
				   max_iter(static_cast<int>(5e5)), tau0(1), verbose(false), restart(true),
				   record_every(30), print_every(100), evaluate_every(30), dataidx(0),
				   save2file(true), print_timing(false), fixed_restart_length(-1)
{
	env.set(GRB_IntParam_OutputFlag, verbose);
}

void Params::set_verbose(const bool &Verbose, const bool &gbVerbose)
{
	verbose = Verbose;
	env.set(GRB_IntParam_OutputFlag, gbVerbose);
}

void Params::load_example()
{
	Eigen::VectorXd c_tmp(5);
	c_tmp << -4, -3, 0, 0, 0;
	c = c_tmp;
	Eigen::VectorXd b_tmp(2);
	b_tmp << 4, 5;
	b = b_tmp;
	Eigen::SparseMatrix<double> A_tmp(2, 5);
	A_tmp.insert(0, 0) = 1;
	A_tmp.insert(0, 2) = 1;
	A_tmp.insert(1, 0) = 1;
	A_tmp.insert(1, 1) = 1;
	A = A_tmp;
	// solution is (4, 1, 0, 0, 0)
}

void Params::load_model(const int &dataidx)
{
	this->dataidx = dataidx;
	this->data_name = Data[dataidx];
	GRBModel model = GRBModel(env, projectpath + "/" + datapath + Data[dataidx] + datasuffix);
	// model.optimize();
	// Get the number of variables in the model.
	int numVars = model.get(GRB_IntAttr_NumVars);

	// Get the number of constraints in the model.
	int numConstraints = model.get(GRB_IntAttr_NumConstrs);

	GRBVar *Vars = model.getVars();
	GRBConstr *Constrs = model.getConstrs();

	// Get the object coefficients from the model.
	c = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_Obj, Vars, numVars), numVars);

	// Get the matrix A, use sparse representation.
	Eigen::SparseMatrix<double, Eigen::RowMajor> A_tmp(numConstraints, numVars);
	std::vector<Eigen::Triplet<double>> triplets;

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

	A_tmp.setFromTriplets(triplets.begin(), triplets.end());
	A = A_tmp;

	// Get the right-hand side vector from the model.
	b = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_RHS,
											  model.getConstrs(), numConstraints),
									numConstraints);

	std::cout << "Model loaded." << std::endl;
}

Eigen::VectorXd &LinearObjectiveTrustRegion(const Eigen::VectorXd &G, const Eigen::VectorXd &L,
											const Eigen::VectorXd &Z, const double &r)
{
	// set l_i to -inf if g_i <= 0
	auto g_idx = G.array() <= 0;
	auto l = g_idx.select(-INFINITY, L);
	auto &z = Z;
	// set g > 0
	auto g = G.cwiseAbs();

	static Eigen::VectorXd z_hat;
	if ((l - z).norm() <= r)
	{
		z_hat = l;
		return z_hat;
	}

	double lambda_mid{-1}, f_low{0}, f_high{0}, f_mid{-1};

	Eigen::VectorXd lambdalist = (z - l).cwiseQuotient(g);
	Eigen::VectorXd lambdalist_sorted = (z - l).cwiseQuotient(g);
	std::sort(lambdalist_sorted.data(), lambdalist_sorted.data() + lambdalist_sorted.size());
	// std::cout << lambdalist_sorted << std::endl;

	// initialize f_low and f_high
	// f_low = (lambdalist.array() <= 0).select(l - z, 0).squaredNorm();
	// f_high = (lambdalist.array() >= INFINITY).select(g, 0).squaredNorm();
	// std::cout << f_low << " " << f_high << std::endl;

	auto size_z = (int)z.size();
	int low{0}, high{size_z - 1}, mid{-1};
	auto r_squared = std::pow(r, 2.0);

	// initialize low and high
	while (low <= size_z - 2 && lambdalist_sorted[low + 1] == 0)
		low++;
	while (high >= 1 && lambdalist_sorted[high - 1] == INFINITY)
		high--;

	while (low + 1 < high)
	{
		mid = (low + high) / 2;
		lambda_mid = lambdalist_sorted[mid];
		// auto idx = (lambdalist_sorted[low] <= lambdalist.array() <= lambdalist_sorted[high]);
		// f_mid = f_low + f_high * std::pow(lambda_mid, 2.0) + idx.select(z_hat - z, 0).squaredNorm();
		f_mid = ((z - lambda_mid * g).cwiseMax(l) - z).squaredNorm();
		if (f_mid < r_squared)
		{
			low = mid;
		}
		else
		{
			high = mid;
		}
	}
	f_low = (lambdalist.array() <= lambdalist_sorted[low]).select(l - z, 0).squaredNorm();
	f_high = (lambdalist.array() >= lambdalist_sorted[high]).select(g, 0).squaredNorm();
	lambda_mid = std::sqrt((r_squared - f_low) / f_high);

	// std::cout << std::endl;
	// std::cout << low << " " << high << std::endl;
	// std::cout << ((z - lambdalist_sorted[low] * g).cwiseMax(l) - z).squaredNorm() << " " << ((z - lambdalist_sorted[high] * g).cwiseMax(l) - z).squaredNorm() << " " << r_squared << std::endl;
	// std::cout << lambda_mid << " " << f_low << " " << f_high << std::endl;
	// std::cout << std::endl;

	z_hat = (z - lambda_mid * G).cwiseMax(L);

	return z_hat;
}

double compute_normalized_duality_gap(const Eigen::VectorXd &x0, const Eigen::VectorXd &y0,
									  const double &r, const Params &p)
{
	int size_x = (int)p.c.rows();
	int size_y = (int)p.b.rows();
	double constant = p.c.dot(x0) - p.b.dot(y0);

	Eigen::VectorXd g(size_x + size_y);
	g << p.c - p.A.transpose() * y0, p.A * x0 - p.b;

	Eigen::VectorXd l = Eigen::VectorXd::Zero(size_x + size_y);
	// set last size_y entries to -inf
	l.tail(size_y) = Eigen::VectorXd::Constant(size_y, -std::numeric_limits<double>::infinity());

	Eigen::VectorXd z0(size_x + size_y);
	z0 << x0, y0;
	auto &z_hat = LinearObjectiveTrustRegion(g, l, z0, r);
	std::cout << std::setprecision(8) << "Trust  gap obj:" << (-z_hat.dot(g) + constant) / r << " ";
	std::cout << "cons vio: ||min(x,0)|| " << z_hat.head(size_x).cwiseMin(0).norm() << " ";
	std::cout << "||z-z0||-r " << (z_hat - z0).norm() - r << " ";
	std::cout << "r: " << r << std::endl;
	if (p.save2file)
	{
		save_obj_residual("trust", (-z_hat.dot(g) + constant) / r, (z_hat - z0).norm() - r);
	}

	return (-g.dot(z_hat) + constant) / r;
}

double compute_normalized_duality_gap(const Eigen::VectorXd &x0, const Eigen::VectorXd &y0,
									  const double &r, const Params &p, const bool use_Gurobi)
{
	int size_x = (int)p.c.rows();
	int size_y = (int)p.b.rows();

	Eigen::VectorXd x_coeff = p.A.transpose() * y0 - p.c;
	Eigen::VectorXd y_coeff = p.b - p.A * x0;

	double constant = p.c.dot(x0) - p.b.dot(y0);

	GRBModel model = GRBModel(p.env);
	// set logfile, use first 6 numbers in r as filename
	std::filesystem::create_directory(projectpath + logpath + "gurobi/");
	std::string filepath_name = projectpath + logpath + "gurobi/" + std::to_string(r).substr(0, 6) + ".log";
	model.set(GRB_StringParam_LogFile, filepath_name);

	// Create variables
	GRBVar *x = new GRBVar[size_x];
	for (int i = 0; i < size_x; i++)
	{
		x[i] = model.addVar(-x0(i), GRB_INFINITY, 0.0, GRB_CONTINUOUS);
	}
	GRBVar *y = new GRBVar[size_y];
	for (int i = 0; i < size_y; i++)
	{
		y[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
	}

	// Create objective
	GRBLinExpr objExpr = GRBLinExpr();
	objExpr.addTerms(y_coeff.data(), y, size_y);
	objExpr.addTerms(x_coeff.data(), x, size_x);

	// Set objective
	// model.setObjective(objExpr, GRB_MAXIMIZE);
	model.setObjective(objExpr + x_coeff.dot(x0) + y_coeff.dot(y0), GRB_MAXIMIZE);

	// Create constraints
	GRBQuadExpr ConstrExpr = GRBQuadExpr();
	Eigen::VectorXd x_quaCoeff = Eigen::VectorXd::Ones(size_x);
	Eigen::VectorXd y_quaCoeff = Eigen::VectorXd::Ones(size_y);
	// Eigen::VectorXd x_linCoeff = -2 * x0;
	// Eigen::VectorXd y_linCoeff = -2 * y0;
	ConstrExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
	ConstrExpr.addTerms(y_quaCoeff.data(), y, y, size_y);
	// ConstrExpr.addTerms(x_linCoeff.data(), x, size_x);
	// ConstrExpr.addTerms(y_linCoeff.data(), y, size_y);
	// ConstrExpr += x0.squaredNorm() + y0.squaredNorm();

	// Add constraints
	model.addQConstr(ConstrExpr, GRB_LESS_EQUAL, r * r);

	model.optimize();

	std::cout << std::setprecision(8) << "Gurobi gap obj:" << (model.get(GRB_DoubleAttr_ObjVal) + constant) / r << " ";
	// get solution
	Eigen::VectorXd x_sol = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);
	Eigen::VectorXd y_sol = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, y, size_y), size_y);
	std::cout << "cons vio: ||min(x,0)|| " << (x_sol + x0).cwiseMin(0).norm() << " ";
	std::cout << "||z-z0||-r " << std::sqrt(x_sol.squaredNorm() + y_sol.squaredNorm()) - r << " ";
	std::cout << "r: " << r << std::endl;
	if (p.save2file)
	{
		save_obj_residual("gurobi", (model.get(GRB_DoubleAttr_ObjVal) + constant) / r, std::sqrt(x_sol.squaredNorm() + y_sol.squaredNorm()) - r);
	}

	std::cout << std::endl;
	// double report_ConstrVio = model.get(GRB_DoubleAttr_ConstrVio);
	if (std::sqrt(x_sol.squaredNorm() + y_sol.squaredNorm()) - r > 1e-3)
	{
		// export_xyr(x0, y0, r);
	}
	else
	{
		// delete the log file
		std::filesystem::remove(filepath_name);
	}
	// std::cout << "model report ConstrVio " << model.get(GRB_DoubleAttr_ConstrVio) << std::endl;

	return (model.get(GRB_DoubleAttr_ObjVal) + constant) / r;
}

void AdaptiveRestarts(Iterates &iter, const Params &p,
					  RecordIterates &record)
{
	if ((iter.count - 1) % p.evaluate_every != 0)
		return;

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
		// ||z_bar^n,t-z^n,0||
		double r1 = std::sqrt((iter.x_bar - iter.cache.x_cur_start).squaredNorm() + (iter.y_bar - iter.cache.y_cur_start).squaredNorm());
		// ||z^n,0-z^n-1,0||

		double duality_gap1 = compute_normalized_duality_gap(iter.x_bar, iter.y_bar, r1, p);
		compute_normalized_duality_gap(iter.x_bar, iter.y_bar, r1, p, 0);

		if (duality_gap1 <= p.beta * iter.cache.gap_cur_prev_start)
		{
			restart = true;
		}
	}

	if (restart == true)
	{
		record.restart_idx.push_back(iter.count - 1);
		if (p.verbose)
		{
			std::cout << "restart at " << iter.count - 1 << std::endl;
			iter.print_iteration_information(p);
		}
		// if ((iter.count - 1) % p.record_every == 0) record.append(iter, p);

		iter.compute_convergence_information(p);
		iter.restart(p);
	}
}

void FixedFrequencyRestart(Iterates &iter, const Params &p,
						   RecordIterates &record)
{
	if ((iter.count - 1) % p.fixed_restart_length == 0)
	{
		iter.compute_convergence_information(p);
		if (p.verbose)
		{
			std::cout << "restart at " << iter.count - 1 << std::endl;
			iter.print_iteration_information(p);
		}
		iter.restart(p);
		if ((iter.count - 1) % p.record_every == 0)
			record.append(iter, p);
	}
}

void GetBestFixedRestartLength(Params &p, RecordIterates (*method)(const Params &))
{
	auto best_gap = INFINITY;
	int best_length = 0;
	p.restart = false;
	std::vector<double> w_candidates;
	for (int i = 1; i < 10; i++)
	{
		w_candidates.push_back(std::pow(4, i));
	}
	for (int i = 0; i < w_candidates.size(); i++)
	{
	}
	return;
}

double PowerIteration(const Eigen::SparseMatrix<double> &A, const bool &verbose = false)
{
	int size = (int)A.rows();
	Eigen::VectorXd u = Eigen::VectorXd::Random(size);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(size);
	double tol = 1e-12;
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
		if (std::abs(lambda - lambda_prev) / std::abs(lambda) < tol)
		{
			if (verbose)
				std::cout << "Power Iteration Converged in " << iter << " iterations." << std::endl;
			break;
		}
		else if (iter >= max_iter)
		{
			if (verbose)
				std::cout << "Maximum Iterations Reached." << std::endl;
			break;
		}
	}
	return lambda;
}

Eigen::VectorXd compute_F(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Params &p)
{
	Eigen::VectorXd F((int)p.c.rows() + (int)p.b.rows());
	Eigen::VectorXd nabla_xL = p.c - p.A.transpose() * y;
	Eigen::VectorXd nabla_yL = p.A * x - p.b;

	F << nabla_xL, nabla_yL;
	return F;
}

double GetOptimalw(Params &p, RecordIterates *(*method)(const Params &))
{
	auto best_kkt_error = INFINITY;
	double best_w = 0, kkt_error;
	p.restart = false;
	std::vector<double> w_candidates;
	RecordIterates *record;
	for (int i = -5; i < 6; i++)
	{
		w_candidates.push_back(std::pow(4, i));
	}
	for (int i = 0; i < w_candidates.size(); i++)
	{
		p.w = w_candidates[i];
		std::cout << "testing w=4^" << i - 5 << std::endl;
		record = method(p);
		kkt_error = record->ConvergeinfoList[record->end_idx - 1].kkt_error;
		if (kkt_error < best_kkt_error)
		{
			best_kkt_error = kkt_error;
			best_w = p.w;
		}
	}
	if (p.verbose)
		std::cout << "the optimal w is: " << best_w << std::endl;
	return best_w;
}

void export_xyr(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double r)
{
	auto path = projectpath + outputpath + "xyr.txt";
	std::ofstream out(path, std::ios::app);
	if (out.is_open())
	{
		out << std::setprecision(15) << x.transpose() << std::endl;
		out << y.transpose() << std::endl;
		out << r << std::endl;
		out.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}

void save_obj_residual(const std::string method, const double obj, const double residual)
{
	auto path = projectpath + outputpath + method + "_obj_residual.txt";
	std::ofstream out(path, std::ios::app);
	if (out.is_open())
	{
		out << std::setprecision(15) << obj << " " << residual << std::endl;
		out.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}