#pragma once
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
    Iterates(const int, const int);
    void update();
};

struct Params
{
    double eta{1e-2}, beta{1e-1};
    int max_iter{10 ^ 2};
    Eigen::VectorXd c;
    Eigen::VectorXd b;
    Eigen::SparseMatrix<double> A;
};

double compute_normalized_duality_gap(const Eigen::VectorXd&, const Eigen::VectorXd&, const Params&);

void load_model(Params&);