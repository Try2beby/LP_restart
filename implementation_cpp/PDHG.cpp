// #include <vector>
#include "gurobi_c++.h"
#include <iostream>

const int M = 100, N = 100;

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

    GRBEnv env = GRBEnv();

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

    // Get the number of variables in the model.
    int numVars = model.get(GRB_IntAttr_NumVars);

    // Get the number of constraints in the model.
    int numConstraints = model.get(GRB_IntAttr_NumConstrs);

    // Get the coefficients from the model.
    double c[] = { model.get(GRB_DoubleAttr_Obj,model.getVars()) };

    // Create a two-dimensional pointer to an array of double values to store the constraint matrix.
    double** A = new double* [numConstraints];
    for (int i = 0; i < numConstraints; i++) {
        A[i] = new double[numVars];
    }

    // Get the constraint matrix from the model.
    //model.get(GRB_DoubleAttr_A, A);

    // Get the right-hand side vector from the model.
    double b[] = { model.get(GRB_DoubleAttr_RHS) };
}

int main()
{
    load_model();
    return 0;
}
