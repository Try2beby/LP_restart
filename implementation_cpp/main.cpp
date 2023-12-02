#include "shared_functions.h"

typedef RecordIterates *(*Method)(Params &);
// define a map from string to function pointer, "PDHG" -> PDHG, "ADMM" -> ADMM
std::map<std::string, Method> method_map = {{"PDHG", PDHG}, {"ADMM", ADMM}};
// define the inverse map
std::map<Method, std::string> inverse_method_map = {{PDHG, "PDHG"}, {ADMM, "ADMM"}};

void process_argument(int argc, char *argv[], Params &p, Method &m);

int main(int argc, char *argv[])
{
	Params p;
	Method m;
	process_argument(argc, argv, p, m);
	auto path = projectpath + outputpath +
				inverse_method_map[m] + "/" + p.data_name + "/";
	if (!std::filesystem::exists(path))
	{
		std::filesystem::create_directories(path);
	}
	// redirect output to file
	std::ofstream out(path + p.outfile_name);
	std::cout.rdbuf(out.rdbuf());
	p.max_iter = 1e12;
	p.max_time = 3600 * 2;
	p.save2file = 0;
	p.print_timing = false;
	p.print_every = 100;
	// p.w = std::pow(4, 2);
	p.set_verbose(1, 0);
	// p.w = std::pow(4, 2);
	// p.load_pagerank();
	p.load_model();

	// std::cout << p.c.transpose() << std::endl;
	// std::cout << p.K << std::endl;
	// std::cout << p.q.transpose() << std::endl;
	// std::cout << p.lb.transpose() << std::endl;
	// std::cout << p.ub.transpose() << std::endl;

	if (p.precondition)
	{
		p.scaling();
	}

	SpMat KKT = p.K * p.K.transpose();
	double sigma_max = std::sqrt(PowerIteration(KKT, 1)); // 1 for verbose
	std::cout << "sigma_max " << sigma_max << std::endl;
	if (p.eta == 0.0)
	{
		p.eta = 0.9 / sigma_max;
	}
	std::cout << "eta " << p.eta << std::endl;
	p.eta_hat = 1 / (p.K.cwiseAbs() * Eigen::VectorXd::Ones(p.K.cols())).maxCoeff();
	std::cout << "eta_hat " << p.eta_hat << std::endl;
	// p.eta_hat = 5e-4;
	auto record = m(p);
	delete record;
	std::cout << "all done" << std::endl;
	// close output file
	out.close();

	return 0;
}

void process_argument(int argc, char *argv[], Params &p, Method &m)
{
	std::map<std::string, std::string> arg_map = {{"adaptive_step_size", "0"},
												  {"restart", "0"},
												  {"primal_weight_update", "0"},
												  {"presolve", "0"},
												  {"scaling", "0"},
												  {"tol", "-8"},
												  {"data_name", "foo"},
												  {"method", "PDHG"},
												  {"fixed_restart_length", "-1"},
												  {"eta", "0"}};

	std::map<std::string, std::string> abr_map = {{"adaptive_step_size", "a"},
												  {"restart", "r"},
												  {"primal_weight_update", "pwu"},
												  {"presolve", "p"},
												  {"scaling", "s"},
												  {"tol", "t"},
												  {"data_name", "d"},
												  {"method", "m"},
												  {"fixed_restart_length", "f"}};

	for (int i = 1; i < argc; i += 2)
	{
		arg_map[argv[i]] = argv[i + 1];
	}
	m = method_map[arg_map["method"]];
	p.adaptive_step_size = std::stoi(arg_map["adaptive_step_size"]);
	p.restart = std::stoi(arg_map["restart"]);
	p.fixed_restart_length = std::stoi(arg_map["fixed_restart_length"]);
	p.primal_weight_update = std::stoi(arg_map["primal_weight_update"]);
	p.precondition = std::stoi(arg_map["scaling"]);
	p.eps = std::pow(10, std::stoi(arg_map["tol"]));
	p.data_name = arg_map["data_name"];
	p.eta = std::stod(arg_map["eta"]);
	// set output file name as restart+primal_weight_update+scaling+tol
	p.outfile_name = abr_map["adaptive_step_size"] + arg_map["adaptive_step_size"] +
					 "_" + abr_map["restart"] + arg_map["restart"] +
					 "_" + abr_map["fixed_restart_length"] + arg_map["fixed_restart_length"] +
					 "_" + abr_map["primal_weight_update"] + arg_map["primal_weight_update"] +
					 "_" + abr_map["scaling"] + arg_map["scaling"] +
					 "_" + abr_map["tol"] + arg_map["tol"] + ".txt";
}