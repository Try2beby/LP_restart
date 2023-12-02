#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <gurobi_c++.h>

// google test
// #include <gtest/gtest.h>

#include "config.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"

// #define EIGEN_USE_MKL_ALL
// #define EIGEN_VECTORIZE_SSE4_2
// #define EIGEN_DONT_PARALLELIZE
// #define EIGEN_USE_BLAS

// #include "eigen3/Eigen/PardisoSupport"

// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>
// #include <boost/serialization/vector.hpp>

const std::string projectpath = PROJECT;
const std::vector<std::string> Data = {"qap10", "qap15", "nug08-3rd", "nug20"};
const std::string cachepath = "cache/";
const std::string cachesuffix = ".txt";
const std::string datapath = "/home/twh/data_manage/";
const std::string pagerankpath = "pagerank/";
const std::string datasuffix = ".mps";
const std::string logpath = "log/";
const std::string outputpath = "output/";
const std::string presolvedpath = "presolved/";

// typedef Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Solver;
typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver;
// typedef Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<double>> Solver;
// BiCGSTAB
// typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> Solver;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatRow;

using namespace std::chrono;

struct Beta
{
	double sufficent{0.1}, necessary{0.9}, artificial{0.5};
	Beta(const double beta_s, const double beta_n, const double beta_a)
		: sufficent(beta_s), necessary(beta_n), artificial(beta_a)
	{
	}
};
struct Cache
{
	Eigen::VectorXd x_prev_start, x_cur_start, y_prev_start, y_cur_start;
	Eigen::VectorXd xU_prev_start, xU_cur_start, xV_prev_start, xV_cur_start;
	double gap_cur_prev_start;
	double mu_c, eta_sum;
};

struct ADMMmodel
{
	GRBModel model_xU, model_xV;
};

struct Convergeinfo
{
	double duality_gap, primal_feasibility_eq, primal_feasibility_ineq, kkt_error, normalized_duality_gap,
		primal_objective, dual_objective, dual_feasibility;
};

class Params
{
public:
	float eta, eta_hat, w, eps, eps_0, theta;
	Beta beta;
	int dataidx, tau0, record_every, print_every, evaluate_every, fixed_restart_length;
	unsigned long long max_iter, max_time;
	int m, m1, m2, n;
	std::string data_name, outfile_name;
	Eigen::VectorXd c, b, q, lb, ub;
	Eigen::VectorXi sense_vec;
	SpMat A, K, D2_cache, D1_cache;
	bool verbose, restart, save2file, print_timing, adaptive_step_size, primal_weight_update, precondition, use_ADMM;
	GRBEnv env;
	Params();
	void init_w();
	void update_w(const Cache &);
	void load_pagerank();
	void scaling();
	void load_model();
	void set_verbose(const bool &, const bool &);
};

class Timer
{
public:
	high_resolution_clock::time_point start_time;
	std::vector<double> time_record;
	Timer();
	float timing();
	void save(const std::string, const Params &, const int);
};

class Iterates
{
public:
	bool use_ADMM, terminate;
	int size_x, size_y, size_z;
	Eigen::VectorXd x, y, x_hat, y_hat, x_bar, y_bar;
	Eigen::VectorXd xU, xV, xU_hat, xV_hat, xU_bar, xV_bar;
	Cache cache;
	int n, t, count;
	high_resolution_clock::time_point time, start_time;
	Iterates(const int &, const int &);
	Iterates(const int &, const int &, const int &);
	void update(const Params &);
	void restart(const Eigen::VectorXd &, const Eigen::VectorXd &);
	void now_time();
	float timing();
	float end();
	Convergeinfo compute_convergence_information(const Params &);
	void compute_primal_objective(const Params &);
	void compute_dual_objective(const Params &);
	Convergeinfo convergeinfo;
	void print_iteration_information(const Params &);
};

class RecordIterates
{
public:
	bool use_ADMM;
	int end_idx;
	std::vector<Iterates> IteratesList;
	std::vector<Convergeinfo> ConvergeinfoList;
	std::vector<int> restart_idx;
	RecordIterates(const int &, const int &, const int &);
	RecordIterates(const int &, const int &, const int &, const int &);
	void append(const Iterates &, const Params &);
	Iterates operator[](const int &);
	void saveConvergeinfo(const std::string, const std::string, const std::string);
	void saveRestart_idx(const std::string, const std::string, const std::string);
};

double compute_normalized_duality_gap(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &, const Params &);
double compute_normalized_duality_gap(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &r, const Params &p, const bool use_Gurobi);
double compute_normalized_duality_gap(const Eigen::VectorXd &x0, const Eigen::VectorXd &y0,
									  const Eigen::VectorXd &x_coeff, const Eigen::VectorXd &y_coeff,
									  const double &r, const Params &p);

Eigen::VectorXd &LinearObjectiveTrustRegion(const Eigen::VectorXd &g, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
											const Eigen::VectorXd &z, const double &r);

void AdaptiveRestarts(Iterates &, Params &, RecordIterates &);
void FixedFrequencyRestart(Iterates &, Params &, RecordIterates &);

double PowerIteration(const Eigen::SparseMatrix<double> &, const bool &);

Eigen::VectorXd compute_F(const Eigen::VectorXd &, const Eigen::VectorXd &, const Params &);

double GetOptimalw(Params &p, RecordIterates *(*method)(const Params &));
void GetBestFixedRestartLength(Params &, RecordIterates (*method)(const Params &));

RecordIterates *ADMM(Params &);
void ADMMStep(Iterates &, const Params &, RecordIterates &, std::vector<GRBModel> &);
void ADMMStep(Iterates &iter, const Params &, RecordIterates &,
			  Solver &);
Eigen::VectorXd update_x(const Eigen::VectorXd &, const double &, const Eigen::VectorXd &,
						 const double &, GRBModel &, const bool &, const int &, const int &);
void generate_update_model(const Params &, std::vector<GRBModel> &);

void PDHGStep(Iterates &, const Params &, RecordIterates &);
RecordIterates *PDHG(Params &);

void EGMStep(Iterates &, const Params &, RecordIterates &);
RecordIterates *EGM(const Params &);

void export_xyr(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double r);
double PDHGnorm(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const int w);
void save_obj_residual(const std::string method, const double obj, const double residual);