#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include "gurobi_c++.h"

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include "eigen3/eigen/core"
#include "eigen3/eigen/sparse"
#include "eigen3/Eigen/PardisoSupport"


using namespace std::chrono;

struct Cache
{
	Eigen::VectorXd z_prev_start, z_cur_start;
};

struct ADMMmodel
{
	GRBModel model_xU, model_xV;
};

struct Convergeinfo
{
	double normalized_duality_gap{ -1 }, kkt_error{ -1 };
};

class Params
{
public:
	float eta, beta, w;
	int max_iter, tau0, record_every, print_every, evaluate_every;
	Eigen::VectorXd c, b;
	Eigen::SparseMatrix<double, Eigen::RowMajor> A;
	bool verbose, restart;
	GRBEnv env;
	Params();
	void load_example();
	void load_model(const std::string&);
	void set_verbose(const bool&, const bool&);
};

class Iterates
{
public:
	bool use_ADMM;
	int size_x, size_y, size_z;
	Eigen::VectorXd z, z_hat, z_bar;
	Cache cache;
	int n, t, count;
	Iterates(const int&, const int&);
	Iterates(const int&, const int&, const int&);
	void update();
	void restart();
	Convergeinfo compute_convergence_information(const Params&);
	Convergeinfo convergeinfo;
	void print_iteration_information(const Params&) const;
	Eigen::VectorXd getx() const;
	Eigen::VectorXd gety() const;
	Eigen::VectorXd getxU() const;
	Eigen::VectorXd getxV() const;
};


class RecordIterates
{
public:
	bool use_ADMM;
	int end_idx;
	std::vector<Iterates> IteratesList;
	std::vector<Convergeinfo> ConvergeinfoList;
	RecordIterates(const int&, const int&, const int&);
	RecordIterates(const int&, const int&, const int&, const int&);
	void append(const Iterates&, const Params&);
	Iterates operator[](const int&);
};


double compute_normalized_duality_gap(const Eigen::VectorXd&, const double&, const Params&);

void AdaptiveRestarts(Iterates&, const Params&, RecordIterates&);
void FixedFrequencyRestart(Iterates&, const Params&,
	RecordIterates&, const int);

double PowerIteration(const Eigen::SparseMatrix<double>&, const bool&);

Eigen::VectorXd compute_F(const Eigen::VectorXd&, const Params&);

double GetOptimalw(Params& p, RecordIterates(*method)(const Params&));
void GetBestFixedRestartLength(Params&, RecordIterates(*method)(const Params&));

RecordIterates ADMM(const Params&);
void ADMMStep(Iterates&, const Params&, RecordIterates&, std::vector<GRBModel>&);
void ADMMStep(Iterates& iter, const Params&, RecordIterates&,
	Eigen::SparseLU<Eigen::SparseMatrix<double>>&);
Eigen::VectorXd update_x(const Eigen::VectorXd&, const double&, const Eigen::VectorXd&,
	const double&, GRBModel&, const bool&, const int&, const int&);
void generate_update_model(const Params&, std::vector<GRBModel>&);

void PDHGStep(Iterates&, const Params&, RecordIterates&);
RecordIterates PDHG(const Params&);

void EGMStep(Iterates&, const Params&, RecordIterates&);
RecordIterates EGM(const Params&);