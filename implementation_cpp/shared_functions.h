#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include "gurobi_c++.h"
#include "eigen3/eigen/core"
#include "eigen3/eigen/sparsecore"

using namespace std::chrono;

class Params
{
public:
	float eta, beta, w;
	int max_iter, tau0, record_every, print_every, evaluate_every;
	Eigen::VectorXd c, b;
	Eigen::SparseMatrix<double, Eigen::RowMajor> A;
	bool verbose, restart;
	GRBEnv env;
	std::vector<GRBConstr> constrs;
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
	int n, t, count;
	Iterates(const int&, const int&);
	Iterates(const int&, const int&, const int&);
	void update();
	void restart();
	double compute_convergence_information(const Params&) const;
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
	std::vector<double> kkt_errorList;
	RecordIterates(const int&, const int&, const int&);
	RecordIterates(const int&, const int&, const int&, const int&);
	void append(const Iterates&, const Params&);
	Iterates operator[](const int&);
};

struct Cache
{
	Eigen::VectorXd z_prev_start, z_cur_start;
};

struct ADMMmodel
{
	GRBModel model_xU, model_xV;
};


double compute_normalized_duality_gap(const Eigen::VectorXd&, const double&, const Params&);

void AdaptiveRestarts(Iterates&, const Params&, RecordIterates&, Cache&);

double PowerIteration(const Eigen::SparseMatrix<double>&, const bool&);

Eigen::VectorXd compute_F(const Eigen::VectorXd&, const Params&);

double GetOptimalw(Params& p, RecordIterates(*method)(const Params&));

void ADMM(const Params&);
void ADMMStep(Iterates&, const Params&, RecordIterates&, std::vector<GRBModel>&);
Eigen::VectorXd update_x(const Eigen::VectorXd&, const double&, const Eigen::VectorXd&,
	const double&, GRBModel&, const bool&, const int&, const int&);
void generate_update_model(const Params&, std::vector<GRBModel>&);

void PDHGStep(Iterates&, const Params&, RecordIterates&);
RecordIterates PDHG(const Params&);

void EGMStep(Iterates&, const Params&, RecordIterates&);
void EGM(const Params&);