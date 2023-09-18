#include <vector>

const int M = 100, N = 100;

struct Iterates
{
    double z[M + N], z_hat[M + N], z_bar[M + N];
    int n, t, count;
};

struct Params
{
    double eta{1e-2}, beta{1e-1};
    int max_iter{1e5};
};

Iterates PrimalDualStep(Iterates z, Params p)
{
    Iterates tmp;
    return tmp;
}

void PrimalDualMethods()
{
    std::vector<Iterates> IteratesList;

    int n{0}, count{0};
    while (true)
    {
        int t{0};
        while (true)
        {
            z, z_hat = PrimalDualStep(z, ETA);
            z_bar = (t / (t + 1)) * z_bar + z_hat / (t + 1);
            t++;
        }
    }
}
