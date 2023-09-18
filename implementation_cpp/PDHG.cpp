

const int MAX_ITER = 1e5;
const double ETA = 1e-2;
const double BETA = 1e-1;

double PrimalDualStep(double z, double eta){

    return 1,2;
}

void PrimalDualMethods()
{   
    double z{0},z_hat{0},z_bar{0};

    int n{0},count{0};
    while (true)
    {   
        int t{0};
        while (true)
        {   
            z,z_hat=PrimalDualStep(z,ETA);
            z_bar=(t/(t+1))*z_bar+z_hat/(t+1);
            t++;

        }
        
        
    }

    
}
