#include "presolve.h"

using namespace Eigen;

struct PresolveInfo
{
    int org_size_x, org_size_y, empty_rows, empty_cols;
    VectorXd var_lb, var_ub;
};

std::set<int> getNonEmptyRows(const SpMat &mat)
{
    std::set<int> nonEmptyRows;
    for (int k = 0; k < mat.outerSize(); ++k)
    {
        for (SpMat::InnerIterator it(mat, k); it; ++it)
        {
            nonEmptyRows.insert(it.row());
        }
    }
    // sort the idx
    std::sort(nonEmptyRows.begin(), nonEmptyRows.end());
    return nonEmptyRows;
}

void presolve(const Params &p, bool transform_bounds = false)
{
}

void remove_empty_rows(const Params &p, PresolveInfo &info)
{
}