#include "shared_functions.h"

Iterates::Iterates(const int Size_x, const int Size_y) : n(0), t(0), count(0)
{
    size_x = Size_x;
    size_y = Size_y;
    size_z = size_x + size_y;
    z = Eigen::VectorXd::Zero(size_z);
    z_hat = Eigen::VectorXd::Zero(size_z);
    z_bar = Eigen::VectorXd::Zero(size_z);
}

void Iterates::update()
{
    z_bar = t * 1.0 / (t + 1) * z_hat + 1.0 / (t + 1) * z_hat;
}

void load_model(Params& p)
{
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env, "data/qap10.mps");
    model.update();

    // Get the number of variables in the model.
    int numVars = model.get(GRB_IntAttr_NumVars);

    // Get the number of constraints in the model.
    int numConstraints = model.get(GRB_IntAttr_NumConstrs);
    // std::cout << numConstraints << std::endl;

    GRBVar* Vars = model.getVars();
    GRBConstr* Constrs = model.getConstrs();

    // Get the object coefficients from the model.
    Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_Obj, Vars, numVars), numVars);

    // Get the matrix A, use sparse representation.
    Eigen::SparseMatrix<double> A(numConstraints, numVars);
    std::vector<Eigen::Triplet<double>> triplets;

    // high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int i = 0; i < numConstraints; i++)
    {
        for (int j = 0; j < numVars; j++)
        {
            double tmp = model.getCoeff(Constrs[i], Vars[j]);
            if (tmp != 0.0)
            {
                triplets.push_back(Eigen::Triplet<double>(i, j, tmp));
            }
        }
    }

    /*high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took" << time_span.count() << " seconds.";*/

    A.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << A.nonZeros()<<std::endl;

    // Get the right-hand side vector from the model.
    // auto* b = model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints);
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(model.get(GRB_DoubleAttr_RHS, model.getConstrs(), numConstraints), numConstraints);

    p.c = c;
    p.b = b;
    p.A = A;
}

double compute_normalized_duality_gap(const Eigen::VectorXd& z0, const Eigen::VectorXd& z1, const Params& p)
{
    auto r = (z0 - z1).norm();

    int size_x = p.c.rows();
    int size_y = p.b.rows();

    Eigen::VectorXd x0 = z0.head(size_x);
    Eigen::VectorXd y0 = z0.tail(size_y);

    Eigen::VectorXd y_coeff = p.b - p.A * x0;
    Eigen::VectorXd x_coeff = y0.transpose() * p.A - p.c.transpose();

    double constant = (double)(p.c.transpose() * x0) - (double)(p.b.transpose() * y0);

    // std::cout << y_coeff << x_coeff << constant << std::endl;

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar* x = model.addVars(size_x, GRB_CONTINUOUS);
    GRBVar* y = new GRBVar[size_y];
    for (int i = 0; i < size_y; i++)
    {
        y[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }

    // Create objective
    GRBLinExpr objExpr = GRBLinExpr();
    objExpr.addTerms(y_coeff.data(), y, size_y);
    objExpr.addTerms(x_coeff.data(), x, size_x);
    objExpr += constant;

    // Set objective
    model.setObjective(objExpr, GRB_MAXIMIZE);

    // Create constraints
    GRBQuadExpr ConstrExpr = GRBQuadExpr();
    Eigen::VectorXd x_quaCoeff = Eigen::VectorXd::Ones(size_x);
    Eigen::VectorXd y_quaCoeff = Eigen::VectorXd::Ones(size_y);
    Eigen::VectorXd x_linCoeff = -2 * x0;
    Eigen::VectorXd y_linCoeff = -2 * y0;
    ConstrExpr.addTerms(x_quaCoeff.data(), x, x, size_x);
    ConstrExpr.addTerms(y_quaCoeff.data(), y, y, size_y);
    ConstrExpr.addTerms(x_linCoeff.data(), x, size_x);
    ConstrExpr.addTerms(y_linCoeff.data(), y, size_y);
    ConstrExpr += x0.squaredNorm() + y0.squaredNorm();

    // Add constraints
    model.addQConstr(ConstrExpr, GRB_LESS_EQUAL, r * r);

    model.optimize();

    /*std::cout<<model.get(GRB_DoubleAttr_ObjVal)<<std::endl;
    std::cout<<x[0].get(GRB_DoubleAttr_X)<<std::endl;
    std::cout<<y[0].get(GRB_DoubleAttr_X)<<std::endl;*/

    return model.get(GRB_DoubleAttr_ObjVal) / r;
}

