#include "shared_functions.h"
#include "presolve.h"

SpMat mat;
std::set<int> result = getNonEmptyRows(mat);
