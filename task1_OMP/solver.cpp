#define _USE_MATH_DEFINES
#include <iostream>
#include <omp.h>
#include "grid.h"
#include "solver.h"

// *ниже (N) означает, что используется N-я формула из условия задания 
// (10): u^0_ijk = phi(xi, yj, zk) = {из (3)} = u|t=0
void compute_u0(const Grid& g, std::vector<double>& u_layer, double t) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i) {
        for (int j = 1; j < g.N; ++j) {
            for (int k = 1; k < g.N; ++k) {
                double x = i * g.h_x;
                double y = j * g.h_y;
                double z = k * g.h_z;
                
                u_layer[g.getLinear(i, j, k)] = analytical_solution(g, x, y, z, t);
            }
        }
    }
}

//(12): u^1_ijk = u^0_ijk + a^2*(a^2 * tau^2 / 2) * laplace(phi(xi, yj, zk)) = {a^2 = 1 + из (3) phi=u|t=0=u0} = u^0_ijk + (a^2 * tau^2 / 2) * laplace(u0)
void compute_u1(const Grid& g, std::vector<double>& u1, const std::vector<double>& u0) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i) {
        for (int j = 1; j < g.N; ++j) {
            for (int k = 1; k < g.N; ++k) {
                u1[g.getLinear(i, j, k)] = u0[g.getLinear(i, j, k)] + TAU * TAU * 0.5 * laplace(g, u0, i, j, k);
            }
        }
    }
}

//считаем следующий u_n по формуле из п.3 условия (почему-то формула без номера):
//(u^{n+1}_ijk - 2u^n_ijk + u^{n-1}_ijk) / tau^2 = a^2 * laplace(u^n), учтём a^2=1 =>
//=> u^{n+1}_ijk = 2u^n_ijk - u^{n-1}_ijk + tau^2 * laplace(u^n)
void compute_u_next(const Grid& g, std::vector<double>& u_next, const std::vector<double>& u_curr, const std::vector<double>& u_prev) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i) {
        for (int j = 1; j < g.N; ++j) {
            for (int k = 1; k < g.N; ++k) {
                u_next[g.getLinear(i, j, k)] = 2 * u_curr[g.getLinear(i, j, k)] - u_prev[g.getLinear(i, j, k)] + TAU * TAU * laplace(g, u_curr, i, j, k);
            }
        }
    }
}

//(7), (8), (9): подсчитываем границы
void set_boundary_conditions(const Grid& g, std::vector<double>& u_layer, double t) {
    //границы по оси X
    #pragma omp parallel for collapse(2)
    for (int j = 0; j <= g.N; ++j) {
        for (int k = 0; k <= g.N; ++k) {
            double y = j * g.h_y;
            double z = k * g.h_z;
            double value = analytical_solution(g, 0.0, y, z, t);
            
            u_layer[g.getLinear(0, j, k)] = value;
            u_layer[g.getLinear(g.N, j, k)] = value;
        }
    }
    
    //границы по оси Y
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < g.N; ++i) {
        for (int k = 0; k <= g.N; ++k) {
            double x = i * g.h_x;
            double z = k * g.h_z;
            double value = analytical_solution(g, x, 0.0, z, t);
            
            u_layer[g.getLinear(i, 0, k)] = value;
            u_layer[g.getLinear(i, g.N, k)] = value;
        }
    }
    
    //границы по оси Z
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < g.N; ++i) {
        for (int j = 1; j < g.N; ++j) {
            double x = i * g.h_x;
            double y = j * g.h_y;
            double value = analytical_solution(g, x, y, 0.0, t);
            
            u_layer[g.getLinear(i, j, 0)] = value;
            u_layer[g.getLinear(i, j, g.N)] = value;
        }
    }
}

double compute_error(const Grid& g, const std::vector<double>& numerical, double t) {
    double max_error = 0.0;
    
    #pragma omp parallel
    {
        double local_max_error = 0.0;
        
        #pragma omp for collapse(3) nowait
        for (int i = 0; i <= g.N; ++i) {
            for (int j = 0; j <= g.N; ++j) {
                for (int k = 0; k <= g.N; ++k) {
                    double x = i * g.h_x;
                    double y = j * g.h_y;
                    double z = k * g.h_z;
                    
                    double error = fabs(numerical[g.getLinear(i, j, k)] - analytical_solution(g, x, y, z, t));
                    if (error > local_max_error) {
                        local_max_error = error;
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            if (local_max_error > max_error) {
                max_error = local_max_error;
            }
        }
    }
    
    return max_error;
}

void print_error_info(double error, int step = -1) {
    std::cout << "\tMax inaccuracy on step " << step << " = " << error << std::endl;
}

void solver_execute(Grid& g,
                   double& inaccuracy_1st, double& inaccuracy_last, double& inaccuracy_max,
                   std::vector<double>& result, double& time, int& threads_num) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads_num);

    int total_nodes = (g.N + 1) * (g.N + 1) * (g.N + 1);
    double start_time, end_time;
    std::vector<std::vector<double>> u = {
        std::vector<double>(total_nodes, 0.0),
        std::vector<double>(total_nodes, 0.0),
        std::vector<double>(total_nodes, 0.0)
    };
    
    std::cout << "Solving equation with " << threads_num << " threads. Computing u0, u1..." << std::endl;
    
    start_time = omp_get_wtime();
    
    //вычисление u0 & u1
    std::cout << "Initializing initial conditions..." << std::endl;

    compute_u0(g, u[0], 0.0);
    set_boundary_conditions(g, u[0], 0.0);
    compute_u1(g, u[1], u[0]);
    set_boundary_conditions(g, u[1], TAU);
    
    inaccuracy_1st = compute_error(g, u[1], TAU);
    inaccuracy_max = inaccuracy_1st;
    
    std::cout << "Steps inaccuracy:" << std::endl;
    print_error_info(inaccuracy_1st, 1);
    

    //вычисление u2, ...
    std::cout << "Running time-stepping algorithm. Computing u2,..." << std::endl;
    for (int step = 2; step < STEPS; ++step) {
        int next = step % 3;
        int curr = (step - 1) % 3;
        int prev = (step - 2) % 3;
        double t = TAU * step;
        
        std::cout << "Processing time step " << step << " (t = " << t << ")" << std::endl;
        set_boundary_conditions(g, u[next], t);
        compute_u_next(g, u[next], u[curr], u[prev]);
        
        double step_max_error = compute_error(g, u[next], t);
        if (step_max_error > inaccuracy_max) {
            inaccuracy_max = step_max_error;
        }
        if (step == STEPS - 1) {
            inaccuracy_last = step_max_error;
        }

        print_error_info(step_max_error, step);
    }
    
    end_time = omp_get_wtime();
    time = end_time - start_time;
    result = u[(STEPS - 1) % 3];

    std::cout << "Computation completed in " << time << " seconds" << std::endl;
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Max inaccuracy: " << inaccuracy_max << std::endl;
    std::cout << "  First step inaccuracy: " << inaccuracy_1st << std::endl;
    std::cout << "  Last step inaccuracy: " << inaccuracy_last << std::endl;
    std::cout << "  Total time: " << time << " seconds" << std::endl;
}
