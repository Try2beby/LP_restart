#include <vector>
#include <iostream>
#include <chrono>

using namespace std::chrono;

#include "gurobi_c++.h"
#include "eigen3/eigen/core"
#include "eigen3/eigen/sparsecore"

const int M = 2, N = 4150;

struct Iterates
{
    Eigen::Matrix<double,M+N,1> z, z_hat, z_bar;
    int n, t, count;
};

struct Params
{
    double eta{1e-2}, beta{1e-1};
    int max_iter{10^2};
    Eigen::VectorXd c;
    Eigen::VectorXd b;
    Eigen::SparseMatrix<double> A;
};

void PrimalDualStep(Iterates &z, const Params &p)
{   

}

void update(Iterates &z)
{
    for (int i = 0; i < M + N; i++)
    {
        z.z_bar[i] = z.t*1.0 / (z.t + 1) * z.z_hat[i] + 1.0 / (z.t + 1) * z.z_hat[i];
    }
}

//void PrimalDualMethods()
//{
//    Iterates z;
//    Params p;
//    std::vector<Iterates> IteratesList(p.max_iter);
//
//    while (true)
//    {
//        while (true)
//        {
//            PrimalDualStep(z, p);
//        }
//    }
//}

GRBModel QPmodel(const Eigen::Matrix<double,M,1> &p,const Params &params,const bool &positive)
{
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar* x=new GRBVar[p.rows()];
    if (positive == true) {
        x = model.addVars(p.rows(), GRB_CONTINUOUS);
    }
    else {
        for (int i = 0; i < p.rows(); i++) {
			x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
		}
    }

    // Create objective 
    GRBQuadExpr quadExpr = GRBQuadExpr();
    for (int i = 0; i < p.rows(); i++) {
        quadExpr.addTerm(p[i], x[i]);
		quadExpr.addTerm(1.0 / (2 * params.eta),x[i],x[i]);
	}
    // Set objective
    model.setObjective(quadExpr, GRB_MINIMIZE);
    
    model.optimize();
    return model;
}

void load_model(Params &p)
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

    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int i = 0; i < numConstraints;i++) {
        for (int j = 0; j < numVars;j++) {
            double tmp = model.getCoeff(Constrs[i], Vars[j]);
            if (tmp != 0.0) {
                triplets.push_back(Eigen::Triplet<double>(i, j, tmp));
            }
        }
    }

    /*high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took" << time_span.count() << " seconds.";*/

    A.setFromTriplets(triplets.begin(), triplets.end());
    //std::cout << A.nonZeros()<<std::endl;
    
    // Get the right-hand side vector from the model.
    //auto* b = model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints);
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints),numConstraints);

    p.c = c;
    p.b = b;
    p.A = A;
}



int main()
{   
    Params p;
    /*load_model(p);
    std::cout << (p.A).nonZeros() << std::endl;*/
    p.eta = 0.5;
    Eigen::Matrix<double, M, 1> p1{2,-4};
    GRBModel model=QPmodel(p1,p,0);
    std::cout << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
    //std::cout << p1.rows();
    return 0;
}
