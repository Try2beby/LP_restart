#include <vector>
#include <iostream>

#include "gurobi_c++.h"
#include "eigen3/eigen/core"
#include "eigen3/eigen/sparsecore"

const int M = 4150, N = 1820;

struct Iterates
{
    double z[M + N], z_hat[M + N], z_bar[M + N];
    int n, t, count;
};

struct Params
{
    double eta{1e-2}, beta{1e-1};
    int max_iter{10^5};
};

void PrimalDualStep(Iterates &z, Params p)
{   

}

void update(Iterates &z)
{
    for (int i = 0; i < M + N; i++)
    {
        z.z_bar[i] = z.t*1.0 / (z.t + 1) * z.z_hat[i] + 1.0 / (z.t + 1) * z.z_hat[i];
    }
}

void PrimalDualMethods()
{
    std::vector<Iterates> IteratesList;
    Iterates z;
    Params p;

    while (true)
    {
        while (true)
        {
            PrimalDualStep(z, p);
        }
    }
}

void load_model()
{
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env, "data/qap10.mps");
    model.update();

    // Get the number of variables in the model.
    int numVars = model.get(GRB_IntAttr_NumVars);

    // Get the number of constraints in the model.
    int numConstraints = model.get(GRB_IntAttr_NumConstrs);
    //std::cout << numConstraints << std::endl;

    GRBVar *Vars = model.getVars();
    GRBConstr* Constrs = model.getConstrs();

    // Get the object coefficients from the model.
    //auto* c = model.get(GRB_DoubleAttr_Obj, model.getVars(), numVars);
    Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_Obj, Vars, numVars),numVars);
    
    // Get the matrix A, use sparse representation.
    Eigen::SparseMatrix<double> A(numConstraints, numVars);
    std::vector<Eigen::Triplet<double>> triplets;

    for (int i = 0; i < numConstraints;i++) {
        for (int j = 0; j < numVars;j++) {
            double tmp = model.getCoeff(Constrs[i], Vars[j]);
            if (tmp != 0.0) {
                triplets.push_back(Eigen::Triplet<double>(i, j, tmp));
            }
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << A.nonZeros()<<std::endl;
    
    // Get the right-hand side vector from the model.
    //auto* b = model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints);
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints),numConstraints);

}


int main()
{
    load_model();
    return 0;
}
