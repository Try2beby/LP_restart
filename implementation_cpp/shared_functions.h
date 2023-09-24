#pragma once
#include <string>
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
	Eigen::VectorXd getx() const;
	Eigen::VectorXd gety() const;
};

class Params
{
private:
	std::string data;

public:
	float eta, beta, w;
	int max_iter, tau0;
	Eigen::VectorXd c;
	Eigen::VectorXd b;
	Eigen::SparseMatrix<double> A;
	bool verbose, restart;
	GRBEnv env;
	Params();
	void load_model(const std::string&);
	void set_verbose(const bool&);
};

double compute_normalized_duality_gap(const Eigen::VectorXd&, const double&, const Params&);

void AdaptiveRestarts(Iterates&, const Params&, std::vector<Iterates>&);

double PowerIteration(const Eigen::SparseMatrix<double>&, const bool&);

Eigen::VectorXd QPmodel(const Eigen::VectorXd&, const Params&, const double&, const bool&);