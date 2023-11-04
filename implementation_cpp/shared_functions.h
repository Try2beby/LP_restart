#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <gurobi_c++.h>
#include "config.h"

// #define EIGEN_USE_MKL_ALL
// #define EIGEN_VECTORIZE_SSE4_2
// #define EIGEN_DONT_PARALLELIZE
#define EIGEN_USE_BLAS

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Sparse"
// #include "eigen3/Eigen/PardisoSupport"

// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>
// #include <boost/serialization/vector.hpp>

const std::string projectpath = PROJECT;
const std::vector<std::string> Data = {"qap10", "qap15", "nug08-3rd", "nug20"};
const std::string cachepath = "cache/";
const std::string cachesuffix = ".txt";
const std::string datapath = "data/";
const std::string datasuffix = ".mps";
const std::string logpath = "log/";
const std::string outputpath = "output/";

// typedef Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Solver;
typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver;
// typedef Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<double>> Solver;
// BiCGSTAB
// typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> Solver;

using namespace std::chrono;

struct Cache
{
	Eigen::VectorXd x_prev_start, x_cur_start, y_prev_start, y_cur_start;
	Eigen::VectorXd xU_prev_start, xU_cur_start, xV_prev_start, xV_cur_start;
	double gap_cur_prev_start;
};

struct ADMMmodel
{
	GRBModel model_xU, model_xV;
};

struct Convergeinfo
{
	double normalized_duality_gap{-1}, kkt_error{-1};
	/*template <class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar& BOOST_SERIALIZATION_NVP(normalized_duality_gap);
		ar& BOOST_SERIALIZATION_NVP(kkt_error);
	}*/
};

class Params
{
public:
	float eta, beta, w, tol;
	int dataidx, max_iter, tau0, record_every, print_every, evaluate_every, fixed_restart_length;
	std::string data_name;
	Eigen::VectorXd c, b;
	Eigen::SparseMatrix<double, Eigen::RowMajor> A;
	bool verbose, restart, save2file, print_timing;
	GRBEnv env;
	Params();
	void load_example();
	void load_model(const int &);
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
	void update(const bool);
	void restart(const Params &);
	void now_time();
	float timing();
	float end();
	Convergeinfo compute_convergence_information(const Params &);
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
	void saveConvergeinfo(const std::string, const int, const std::string);
	void saveRestart_idx(const std::string, const int, const std::string);
};

double compute_normalized_duality_gap(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &, const Params &);
double compute_normalized_duality_gap(const Eigen::VectorXd &, const Eigen::VectorXd &, const double &r, const Params &p, const bool use_Gurobi);
Eigen::VectorXd &LinearObjectiveTrustRegion(const Eigen::VectorXd &g, const Eigen::VectorXd &l,
											const Eigen::VectorXd &z, const double &r);

void AdaptiveRestarts(Iterates &, const Params &, RecordIterates &);
void FixedFrequencyRestart(Iterates &, const Params &, RecordIterates &);

double PowerIteration(const Eigen::SparseMatrix<double> &, const bool &);

Eigen::VectorXd compute_F(const Eigen::VectorXd &, const Eigen::VectorXd &, const Params &);

double GetOptimalw(Params &p, RecordIterates *(*method)(const Params &));
void GetBestFixedRestartLength(Params &, RecordIterates (*method)(const Params &));

RecordIterates *ADMM(const Params &);
void ADMMStep(Iterates &, const Params &, RecordIterates &, std::vector<GRBModel> &);
void ADMMStep(Iterates &iter, const Params &, RecordIterates &,
			  Solver &);
Eigen::VectorXd update_x(const Eigen::VectorXd &, const double &, const Eigen::VectorXd &,
						 const double &, GRBModel &, const bool &, const int &, const int &);
void generate_update_model(const Params &, std::vector<GRBModel> &);

void PDHGStep(Iterates &, const Params &, RecordIterates &);
RecordIterates *PDHG(const Params &);

void EGMStep(Iterates &, const Params &, RecordIterates &);
RecordIterates *EGM(const Params &);

void export_xyr(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double r);
void save_obj_residual(const std::string method, const double obj, const double residual);