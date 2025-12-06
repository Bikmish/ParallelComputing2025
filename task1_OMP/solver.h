#ifndef SOLVER_H
#define SOLVER_H
#include <vector>
#include <cmath>

#define STEPS 30
#define TAU 0.00001

inline double analytical_solution(const Grid& g, const double& x, const double& y, const double& z, const double& t) {
    double at = M_PI * std::sqrt((4.0 / (g.Lx * g.Lx)) + (16.0 / (g.Ly * g.Ly)) + (36.0 / (g.Lz * g.Lz)));
    return std::sin((2.0 * M_PI * x) / g.Lx) *
           std::sin((4.0 * M_PI * y) / g.Ly) *
           std::sin((6.0 * M_PI * z) / g.Lz) *
           std::cos(at * t);
}

inline int get_neighbor_idx(int coord, int delta, int N) {
    int new_coord = coord + delta;
    if (new_coord < 0) return N - 1;
    if (new_coord > N) return 1;
    return new_coord;
}

inline double laplace(const Grid& g, const std::vector<double>& ui, const int& i, const int& j, const int& k) {
    const int N = g.N;
    const int center_idx = g.getLinear(i, j, k);
    const double center_val = ui[center_idx];
    
    double deriv_x = (ui[g.getLinear(get_neighbor_idx(i, -1, N), j, k)] - 2 * center_val + 
                     ui[g.getLinear(get_neighbor_idx(i, 1, N), j, k)]) / (g.h_x * g.h_x);
    
    double deriv_y = (ui[g.getLinear(i, get_neighbor_idx(j, -1, N), k)] - 2 * center_val + 
                     ui[g.getLinear(i, get_neighbor_idx(j, 1, N), k)]) / (g.h_y * g.h_y);
    
    double deriv_z = (ui[g.getLinear(i, j, get_neighbor_idx(k, -1, N))] - 2 * center_val + 
                     ui[g.getLinear(i, j, get_neighbor_idx(k, 1, N))]) / (g.h_z * g.h_z);
    
    return deriv_x + deriv_y + deriv_z;
}

void set_boundary_conditions(const Grid& g, std::vector<double>& u_layer, double t);

void compute_u0(const Grid& g, std::vector<double>& u_layer, double t);

void compute_u1(const Grid& g, std::vector<double>& u1, const std::vector<double>& u0);

void compute_u_next(const Grid& g, std::vector<double>& u_next, const std::vector<double>& u_curr, const std::vector<double>& u_prev);

double compute_error(const Grid& g, const std::vector<double>& numerical, double t);

void print_error_info(double error, int step);

void solver_execute(Grid& grid, double& inaccuracy_1st, double& inaccuracy_last, double& inaccuracy_max, std::vector<double>& result, double& time, int& threads_num);

#endif
