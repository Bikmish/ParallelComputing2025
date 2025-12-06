#include <iostream>
#include <fstream>
#include "grid.h"
#include "solver.h"
#define _USE_MATH_DEFINES

int main(int argc, char* argv[]) {
    int N, num_threads = 0;
    double Lx, Ly, Lz;
    double time, inaccuracy_1st, inaccuracy_last, inaccuracy_max = -1;
    std::vector<double> result;

    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " expects either 3 or 5 parameters:\n"
                  << "\t- 3 parameters: N L_val num_threads - grid size & general bound for dimensions (Lx=Ly=Lz=L_val) & number of threads\n"
                  << "\t- 5 parameters: N Lx Ly Lz - grid size & custom values for every dimension & number of threads" << std::endl;
        return 1;
    }

    N = std::stoi(argv[1]);
    if (argc == 4) {
        double L_val = (std::string(argv[2]) == "Pi") ? M_PI : std::stod(argv[2]);
        num_threads = std::stoi(argv[3]);
        Lx = Ly = Lz = L_val;
        std::cout << "Using uniform dimensions: Lx = Ly = Lz = " << L_val << " with " << num_threads << " threads" << std::endl;
    } else { //argc == 6
        Lx = (std::string(argv[2]) == "Pi") ? M_PI : std::stod(argv[2]);
        Ly = (std::string(argv[3]) == "Pi") ? M_PI : std::stod(argv[3]);
        Lz = (std::string(argv[4]) == "Pi") ? M_PI : std::stod(argv[4]);
        num_threads = std::stoi(argv[5]);
        std::cout << "Using custom dimensions: Lx = " << Lx 
                  << ", Ly = " << Ly 
                  << ", Lz = " << Lz << "with" << num_threads << "threads" << std::endl;
    }

    if (N <= 0 || Lx <= 0 || Ly <= 0 || Lz <= 0 || num_threads < 0) {
        std::cerr << "Error: N, Lx, Ly, Lz must be positive and num_threads must be non-negative!" << std::endl;
        return 1;
    }

    Grid grid = Grid(N, Lx, Ly, Lz);
    solver_execute(grid, inaccuracy_1st, inaccuracy_last, inaccuracy_max, result, time, num_threads);

    std::string fname = "stats/" + std::to_string(grid.N) + "_" + std::to_string((int)grid.Lx) + "_" + std::to_string((int)grid.Ly) + "_" + std::to_string((int)grid.Lz) + "_" + std::to_string(num_threads) + ".txt";  
    std::ofstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Cannot write to file " << fname << "!" << std::endl;
        return 1;
    }

    file << "Summary:" << std::endl;
    file << "  Max inaccuracy: " << inaccuracy_max << std::endl;
    file << "  First step inaccuracy: " << inaccuracy_1st << std::endl;
    file << "  Last step inaccuracy: " << inaccuracy_last << std::endl;
    file << "  Total time: " << time << " seconds" << std::endl;
    return 0;
}
