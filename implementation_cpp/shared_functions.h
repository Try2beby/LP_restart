#pragma once
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <chrono>
#include "gurobi_c++.h"
#include "eigen3/eigen/core"
#include "eigen3/eigen/sparsecore"

class Iterates
{
private:
	int size_x, size_y, size_z;

public:
	Eigen::VectorXd z, z_hat, z_bar;
	int n, t, count;
	Iterates(const int&, const int&);
	void update();
	void restart();
	Eigen::VectorXd getx() const;
	Eigen::VectorXd gety() const;
};

class RecordIterates
{
private:
	int end_idx;
	std::vector<Iterates> IteratesList;

public:
	RecordIterates(const int&, const int&, const int&);
	void append(const Iterates&);
	Iterates operator[](const int&);
};

struct Cache
{
	Eigen::VectorXd z_prev_start, z_cur_start;
};

class Params
{
public:
	float eta, beta, w;
	int max_iter, tau0, record_every, print_every, evaluate_every;
	Eigen::VectorXd c, b;
	Eigen::SparseMatrix<double> A;
	bool verbose, restart;
	GRBEnv env;
	Params();
	void load_model(const std::string&);
	void set_verbose(const bool&);
};

double compute_normalized_duality_gap(const Eigen::VectorXd&, const double&, const Params&);

void AdaptiveRestarts(Iterates&, const Params&, RecordIterates&, Cache&);

double PowerIteration(const Eigen::SparseMatrix<double>&, const bool&);

Eigen::VectorXd QPmodel(const Eigen::VectorXd&, const Params&, const double&, const bool&);

void print_iteration_information(const Iterates&, const Params&);