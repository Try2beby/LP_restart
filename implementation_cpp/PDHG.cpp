#include <vector>
#include <iostream>
#include <chrono>

using namespace std::chrono;

#include "shared_functions.h"

GRBModel QPmodel(const Eigen::VectorXd&, const Params&, const bool&);

int main()
{
    using std::cout, std::endl;
    Params p;

    // test load_model
    load_model(p);
    cout << (p.A).nonZeros() << endl;

    // test QPmodel
    /*p.eta = 0.5;
    Eigen::Matrix<double, 2, 1> p1{2,-4};
    GRBModel model=QPmodel(p1,p,0);
    cout << model.get(GRB_DoubleAttr_ObjVal) << endl;*/

    // test compute_normalized_duality_gap
    /*Eigen::Vector2d z0(0, 0);
    Eigen::Vector2d z1(1, 0);
    p.b = Eigen::VectorXd::Ones(1);
    p.c = -1 * Eigen::VectorXd::Ones(1);
    Eigen::SparseMatrix<double> A(1, 1);
    A.insert(0, 0) = 1;
    p.A = A;
    compute_normalized_duality_gap(z0, z1, p);*/


    return 0;
}

void PrimalDualStep(Iterates& z, const Params& p)
{
}


// void PrimalDualMethods()
//{
//     Iterates z;
//     Params p;
//     std::vector<Iterates> IteratesList(p.max_iter);
//
//     while (true)
//     {
//         while (true)
//         {
//             PrimalDualStep(z, p);
//         }
//     }
// }

GRBModel QPmodel(const Eigen::VectorXd& p, const Params& params, const bool& positive)
{
    int size_x = p.rows();

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar* x;
    if (positive == true)
    {
        x = model.addVars(size_x, GRB_CONTINUOUS);
    }
    else
    {
        x = new GRBVar[size_x];
        for (int i = 0; i < size_x; i++)
        {
            x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        }
    }

    // Create objective
    GRBQuadExpr objExpr = GRBQuadExpr();
    Eigen::VectorXd x_quaCoeff = (1 / (2 * params.eta)) * Eigen::VectorXd::Ones(p.rows());
    objExpr.addTerms(x_quaCoeff.data(), x, x, p.rows());
    objExpr.addTerms(p.data(), x, p.rows());

    // Set objective
    model.setObjective(objExpr, GRB_MINIMIZE);

    model.optimize();
    return model;
}
