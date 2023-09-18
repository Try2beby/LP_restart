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

void PrimalDualStep(Iterates &z, Params p)
{
}

void update(Iterates &z)
{
    for (int i = 0; i < M + N; i++)
    {
        z.z_bar[i] = z.t / (z.t + 1) * z.z_hat[i] + 1 / (z.t + 1) * z.z_hat[i];
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
