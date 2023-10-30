#include "shared_functions.h"

namespace utils
{
	void read_txt(SpMat &A, const std::string filename)
	{
		// read a eigen matrix from txt file
		std::ifstream in(filename);
		std::vector<Eigen::Triplet<double>> triplets;

		if (in.is_open())
		{
			int row, col;
			double value;
			while (in >> row >> col >> value)
			{
				triplets.push_back(Eigen::Triplet<double>(row, col, value));
			}
			A.setFromTriplets(triplets.begin(), triplets.end());
		}
		else
		{
			std::cout << "File not found!" << std::endl;
			exit(1);
		}
		// std::cout << A.rows() << " " << A.cols() << std::endl;
		// std::cout << A.nonZeros() << std::endl;
	}
}

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
	y = (1.0 / size_y) * Eigen::VectorXd::Ones(size_y);
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

void Iterates::update(const Params &p)
{
	if (use_ADMM == false)
	{
		cache.eta_sum += p.eta;
		// x_bar = 1.0 / cache.eta_sum * (x_bar + p.eta * x);
		x_bar = t * 1.0 / (t + 1) * x_bar + 1.0 / (t + 1) * x_hat;

		if (not p.restart)
		{
			cache.x_prev_start = cache.x_cur_start;
			cache.x_cur_start = this->x;
		}
	}
	else
	{
		xU_bar = t * 1.0 / (t + 1) * xU_bar + 1.0 / (t + 1) * xU_hat;
		xV_bar = t * 1.0 / (t + 1) * xV_bar + 1.0 / (t + 1) * xV_hat;
		if (not p.restart)
		{
			cache.xU_prev_start = cache.xU_cur_start;
			cache.xU_cur_start = this->xU;
			cache.xV_prev_start = cache.xV_cur_start;
			cache.xV_cur_start = this->xV;
		}
	}
	// y_bar = 1.0 / cache.eta_sum * (y_bar + p.eta * y);
	y_bar = t * 1.0 / (t + 1) * y_bar + 1.0 / (t + 1) * y_hat;
	if (not p.restart)
	{
		cache.y_prev_start = cache.y_cur_start;
		cache.y_cur_start = this->y;
	}
	t++;
	count++;
}

void Iterates::restart(const Eigen::VectorXd &x_c, const Eigen::VectorXd &y_c)
{
	n++;
	t = 0;
	count++;
	if (use_ADMM == false)
	{
		x = x_c;
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
	y = y_c;
	cache.y_prev_start = cache.y_cur_start;
	cache.y_cur_start = this->y;
}

Convergeinfo Iterates::compute_convergence_information(const Params &p)
{
	Eigen::VectorXd mutiplier, temp;

	this->convergeinfo.duality_gap = std::abs(p.q.dot(y) - p.c.dot(x)) / (1 + std::abs(p.q.dot(y)) + std::abs(p.c.dot(x)));

	temp = p.q - p.K * x;
	temp.segment(0, p.m1) = temp.segment(0, p.m1).cwiseMax(0);
	this->convergeinfo.primal_feasibility = temp.norm() / (1 + p.q.norm());

	temp = p.c - p.K.transpose() * y;
	mutiplier = temp.cwiseMax(0);
	this->convergeinfo.dual_feasibility = (temp - mutiplier).norm() / (1 + p.c.norm());

	// if (std::abs(this->convergeinfo.normalized_duality_gap) < p.tol || this->convergeinfo.kkt_error < p.tol)
	if (this->convergeinfo.duality_gap < p.eps && this->convergeinfo.primal_feasibility < p.eps && this->convergeinfo.dual_feasibility < p.eps)
	{
		this->terminate = true;
		auto duration = this->end();
		std::cout << "Iteration terminates at " << this->count - 1 << ", takes " << duration << "s" << std::endl;
		// print first 10 elements of x, set precision
		std::cout << std::setprecision(8) << "x: " << x.head(10).transpose() << std::endl;
		std::cout << std::endl;
	}
	return this->convergeinfo;
}

void Iterates::print_iteration_information(const Params &p)
{
	std::cout << "Iteration " << count - 1 << ", ";
	std::cout << "duality_gap: " << this->convergeinfo.duality_gap << std::endl;
	std::cout << "primal_feasibility: " << this->convergeinfo.primal_feasibility << std::endl;
	std::cout << "dual_feasibility: " << this->convergeinfo.dual_feasibility << std::endl;
	std::cout << "x.sum " << x.sum() << std::endl;
	std::cout << "x_org.sum " << (p.D2_cache * x).sum() << std::endl;
	// std::cout << x.norm() << " " << y.norm() << std::endl;
	// std::cout << p.eta << " " << p.eta_hat << std::endl;

	if (use_ADMM == false)
	{
		std::cout << "obj: " << p.c.dot(x) + p.q.dot(y) - y.transpose() * p.K * x << std::endl;
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
	// std::vector<Iterates> aIteratesList(Size_record, iter);
	std::vector<Convergeinfo> aConvergeinfoList(Size_record);
	// IteratesList = aIteratesList;
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
	// IteratesList[end_idx] = iter;
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
			ofs << std::setprecision(10) << obj.duality_gap << " " << obj.primal_feasibility << " " << obj.dual_feasibility << std::endl;
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

Params::Params() : env(GRBEnv()), eta(0), eta_hat{1e-1}, w(1), eps(1e-8), eps_0(1e-7),
				   beta(0.9, 0.1, 0.5), theta(0.5),
				   max_iter(static_cast<int>(5e5)), tau0(1), verbose(false), restart(true),
				   record_every(40), print_every(100), evaluate_every(40), dataidx(0),
				   save2file(true), print_timing(false), fixed_restart_length(-1)
{
	env.set(GRB_IntParam_OutputFlag, verbose);
	this->init_w();
}

void Params::init_w()
{
	double c_norm = this->c.norm(), q_norm = this->q.norm();
	if (c_norm > eps_0 && q_norm > eps_0)
	{
		this->w = c_norm / q_norm;
	}
}

void Params::update_w(const Cache &cache)
{
	double delta_x = (cache.x_prev_start - cache.x_cur_start).norm();
	double delta_y = (cache.y_prev_start - cache.y_cur_start).norm();
	if (delta_x > eps_0 && delta_y > eps_0)
	{
		this->w = std::exp(theta * std::log(delta_y / delta_x) + (1 - theta) * std::log(this->w));
	}
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

void Params::load_pagerank()
{
	n = 1e5;
	double lambda = 0.85;
	SpMat A_tmp(n + 1, n);
	utils::read_txt(A_tmp, projectpath + "/" + datapath + "graph_1e5.txt");
	this->A = A_tmp;
	this->q = ((1 - lambda) / n) * Eigen::VectorXd::Ones(n + 1);
	this->q[n] = 1;
	this->c = Eigen::VectorXd::Zero(n);

	// set K^T to be (I_n - lambda * A^T, 1_n)
	SpMat K_tmp(n + 1, n);
	for (int i = 0; i < n; i++)
	{
		K_tmp.insert(i, i) = 1;
	}
	K_tmp = K_tmp - lambda * A_tmp;
	for (int i = 0; i < n; i++)
	{
		K_tmp.insert(n, i) = 1;
	}
	K = K_tmp;

	this->m1 = n;
	this->m2 = 1;

	// print shape and nnz of K
	std::cout << "K shape: " << K.rows() << " " << K.cols() << std::endl;
	std::cout << "K nnz: " << K.nonZeros() << std::endl;
	std::cout << "PageRank model loaded." << std::endl;
}

void Params::scaling()
{
	// Ruiz scaling
	SpMat D1(m1 + m2, m1 + m2), D2(n, n), temp;
	SpMat D1cache(m1 + m2, m1 + m2), D2cache(n, n);
	for (int i = 0; i < m1 + m2; i++)
	{
		D1cache.insert(i, i) = 1;
	}
	for (int i = 0; i < n; i++)
	{
		D2cache.insert(i, i) = 1;
	}
	Eigen::VectorXd temp_vec;

	for (int ii = 0; ii < 10; ii++)
	{
		for (int j = 0; j < K.outerSize(); ++j)
		{
			double max_norm = 0;
			for (Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator it(K, j); it; ++it)
			{
				max_norm = std::max(max_norm, std::abs(it.value()));
			}
			D2.coeffRef(j, j) = 1.0 / std::sqrt(max_norm);
			D2cache.coeffRef(j, j) *= 1.0 / std::sqrt(max_norm);
		}
		temp = K.transpose();
		for (int j = 0; j < temp.outerSize(); ++j)
		{
			double max_norm = 0;
			for (Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator it(temp, j); it; ++it)
			{
				max_norm = std::max(max_norm, std::abs(it.value()));
			}
			D1.coeffRef(j, j) = 1.0 / std::sqrt(max_norm);
			D1cache.coeffRef(j, j) *= 1.0 / std::sqrt(max_norm);
		}
		K = D1 * K * D2;
		c = D2 * c;
		q = D1 * q;
	}

	// Pock Chambolle scaling
	temp_vec = K.cwiseAbs() * Eigen::VectorXd::Ones(K.cols());
	for (int i = 0; i < temp_vec.size(); i++)
	{
		D1.coeffRef(i, i) = 1.0 / std::sqrt(temp_vec(i));
		D1cache.coeffRef(i, i) *= 1.0 / std::sqrt(temp_vec(i));
	}
	temp_vec = K.transpose().cwiseAbs() * Eigen::VectorXd::Ones(K.rows());
	for (int i = 0; i < temp_vec.size(); i++)
	{
		D2.coeffRef(i, i) = 1.0 / std::sqrt(temp_vec(i));
		D2cache.coeffRef(i, i) *= 1.0 / std::sqrt(temp_vec(i));
	}

	this->D1_cache = D1cache;
	this->D2_cache = D2cache;

	K = D1 * K * D2;
	c = D2 * c;
	q = D1 * q;
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
	int size_y = (int)p.q.rows();
	double constant = p.c.dot(x0) - p.q.dot(y0);

	Eigen::VectorXd g(size_x + size_y);
	g << p.c - p.K.transpose() * y0, p.K * x0 - p.q;

	Eigen::VectorXd l = Eigen::VectorXd::Zero(size_x + size_y);
	// set last size_y entries to -inf
	l.tail(size_y) = Eigen::VectorXd::Constant(size_y, -std::numeric_limits<double>::infinity());

	Eigen::VectorXd z0(size_x + size_y);
	z0 << x0, y0;
	auto &z_hat = LinearObjectiveTrustRegion(g, l, z0, r);

	// std::cout << std::setprecision(8) << "Trust  gap obj:" << (-z_hat.dot(g) + constant) / r << " ";
	// std::cout << std::setprecision(16) << "cons vio: ||min(x,0)|| " << z_hat.head(size_x).cwiseMin(0).norm() << " ";
	// std::cout << std::setprecision(16) << "max(||z-z0||-r,0) " << std::max((z_hat - z0).norm() - r, 0.0) << " ";
	// std::cout << std::setprecision(8) << "r: " << r << std::endl;

	return (-g.dot(z_hat) + constant) / r;
}

double compute_normalized_duality_gap(const Eigen::VectorXd &x0, const Eigen::VectorXd &y0,
									  const double &r, const Params &p, const bool use_Gurobi)
{
	int size_x = (int)p.c.rows();
	int size_y = (int)p.b.rows();

	Eigen::VectorXd y_coeff = p.b - p.A * x0;
	Eigen::VectorXd x_coeff = y0.transpose() * p.A - p.c.transpose();

	double constant = p.c.dot(x0) - p.b.dot(y0);

	GRBModel model = GRBModel(p.env);
	// set logfile, use first 6 numbers in r as filename
	std::filesystem::create_directory(projectpath + logpath + "gurobi/");
	model.set(GRB_StringParam_LogFile, projectpath + logpath + "gurobi/" + std::to_string(r).substr(0, 6) + ".log");

	// Create variables
	GRBVar *x = model.addVars(size_x, GRB_CONTINUOUS);
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

	// std::cout << std::setprecision(8) << "Gurobi gap obj:" << (model.get(GRB_DoubleAttr_ObjVal) + constant) / r << " ";
	// get solution
	// Eigen::VectorXd x_sol = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, x, size_x), size_x);
	// Eigen::VectorXd y_sol = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_X, y, size_y), size_y);
	// std::cout << std::setprecision(16) << "cons vio: ||min(x,0)|| " << x_sol.cwiseMin(0).norm() << " ";
	// std::cout << std::setprecision(16) << "max(||z-z0||-r,0) " << std::max(std::sqrt((x_sol - x0).squaredNorm() + (y_sol - y0).squaredNorm()) - r, 0.0) << " ";
	// std::cout << std::setprecision(8) << "r: " << r << std::endl;
	// std::cout << std::setprecision(16) << "||z-z0||-r " << std::sqrt((x_sol - x0).squaredNorm() + (y_sol - y0).squaredNorm()) - r << " ";
	// double report_ConstrVio = model.get(GRB_DoubleAttr_ConstrVio);
	// if (report_ConstrVio > 1e-1)
	// {
	// 	export_xyr(x0, y0, r);
	// }
	// std::cout << std::setprecision(16) << "model report ConstrVio " << model.get(GRB_DoubleAttr_ConstrVio) << std::endl;

	return (model.get(GRB_DoubleAttr_ObjVal) + constant) / r;
}

void AdaptiveRestarts(Iterates &iter, Params &p,
					  RecordIterates &record)
{
	if ((iter.count - 1) % p.evaluate_every != 0)
		return;

	bool restart = false;
	if (iter.t > p.beta.artificial * iter.count)
	{
		restart = true;
	}
	double r1, r2, mu_1, mu_2, mu_c;
	Eigen::VectorXd x_c, y_c;
	// ||z^n,t-z^n,0||
	r1 = PDHGnorm(iter.x - iter.cache.x_cur_start, iter.y - iter.cache.y_cur_start, p.w);
	// mu_1 = std::pow(compute_normalized_duality_gap(iter.x, iter.y, r1, p), 1.0 * iter.n);
	mu_1 = compute_normalized_duality_gap(iter.x, iter.y, r1, p);

	// ||z_bar^n,t-z^n,0||
	r2 = PDHGnorm(iter.x_bar - iter.cache.x_cur_start, iter.y_bar - iter.cache.y_cur_start, p.w);
	// mu_2 = std::pow(compute_normalized_duality_gap(iter.x_bar, iter.y_bar, r2, p), 1.0 * iter.n);
	mu_2 = compute_normalized_duality_gap(iter.x_bar, iter.y_bar, r2, p);

	if (mu_1 < mu_2)
	{
		x_c = iter.x;
		y_c = iter.y;
	}
	else
	{
		x_c = iter.x_bar;
		y_c = iter.y_bar;
	}
	mu_c = std::min(mu_1, mu_2);

	double r3, mu_3{0};
	// ||z^n,0-z^n-1,0||
	if (iter.n >= 1)
	{
		r3 = PDHGnorm(iter.cache.x_prev_start - iter.cache.x_cur_start, iter.cache.y_prev_start - iter.cache.y_cur_start, p.w);
		// mu_3 = std::pow(compute_normalized_duality_gap(iter.cache.x_cur_start, iter.cache.y_cur_start, r3, p), 1.0 * iter.n);
		mu_3 = compute_normalized_duality_gap(iter.cache.x_cur_start, iter.cache.y_cur_start, r3, p);
	}

	// ||z_c - z^n,0||
	// r4 = PDHGnorm(iter.cache.x_c - iter.cache.x_cur_start, iter.cache.y_c - iter.cache.y_cur_start, p.w);
	// mu_4 = std::pow(compute_normalized_duality_gap(iter.cache.x_cur_start, iter.cache.y_cur_start, r3, p), 1.0 * iter.n);
	if (mu_c <= p.beta.sufficent * mu_3)
	{
		restart = true;
	}
	// else if (mu_c <= p.beta.necessary * mu_3 && mu_c > iter.cache.mu_c)
	// {
	// 	restart = true;
	// }
	iter.cache.mu_c = mu_c;

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
		iter.restart(x_c, y_c);
		if (p.primal_weight_update)
		{
			p.update_w(iter.cache);
		}
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
		// iter.restart();
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
		kkt_error = record->ConvergeinfoList[record->end_idx - 1].dual_feasibility;
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
		out << std::setprecision(15) << y.transpose() << std::endl;
		out << std::setprecision(15) << r << std::endl;
		out.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;
}
