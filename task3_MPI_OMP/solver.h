#ifndef SOLVER_H
#define SOLVER_H

#include "utils.h"
#include <cmath>

#define TAU 0.00001
#define STEPS 20
#define TIME_LAYERS_WINDOW_SIZE 3


inline double analytical_solution(double x, double y, double z, double t, double Lx, double Ly, double Lz) {
    double at = M_PI * sqrt(4.0/(Lx*Lx) + 16.0/(Ly*Ly) + 36.0/(Lz*Lz));
    return sin(2 * M_PI * x / Lx) * 
           sin(4 * M_PI * y / Ly) * 
           sin(6 * M_PI * z / Lz) *
           cos(at * t);
}

extern double hx, hy, hz;
inline double calculate_second_derivative(
    const double data[], 
    int i, int j, int k,
    int di, int dj, int dk,
    double h)
{
    int p_curr = index(i, j, k);
    int p_prev = index(i - di, j - dj, k - dk);
    int p_next = index(i + di, j + dj, k + dk);

    return (data[p_prev] - 2 * data[p_curr] + data[p_next]) / (h * h);
}

inline double laplace(const double data[], int i, int j, int k) {
    double val = 0;

    //ось X
    val += calculate_second_derivative(data, i, j, k, 1, 0, 0, hx);

    //ось Y
    val += calculate_second_derivative(data, i, j, k, 0, 1, 0, hy);
    
    //ось Z
    val += calculate_second_derivative(data, i, j, k, 0, 0, 1, hz);

    return val;
}

void compute_error(ErrorInfo* p_error, const double *data, int t, int i_min, int j_min, int k_min, double Lx, double Ly, double Lz) {
    double mse = 0;
    double max = 0;

#pragma omp parallel for reduction(+:mse)
    for (int p = 0; p < dx * dy * dz; p++) {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        // пропускаем обменные области
        if (i == 0 || j == 0 || k == 0 || i == dx - 1 || j == dy - 1 || k == dz - 1)
            continue;

        double u_true = analytical_solution((i_min + i - 1) * hx, (j_min + j - 1) * hy, (k_min + k - 1) * hz, t * TAU, Lx, Ly, Lz);
        double u_pred = data[p];

        mse += pow(u_true - u_pred, 2);

        #pragma omp critical
        {
            double u_abs = fabs(u_true - u_pred);
            if (u_abs > max)
                max = u_abs;
        }
    }

    p_error->max = max;
    p_error->mse = mse;
}

void set_boundary_conditions(double *data, bool is_first[3], bool is_last[3])
{
        /*   Примечание:
        * левая граница (i=0) должна равняться ПРАВОЙ внутренней границе
        * правая граница (i=dx-1) должна равняться ЛЕВОЙ внутренней границе
        */ 
#pragma omp parallel
    {
        //ось X
#pragma omp for nowait
        for (int p = 0; p < dy * dz; p++) {
            int j = p % dy;
            int k = (p / dy) % dz;
            
            if (is_first[0]) {
                data[index(0, j, k)] = data[index(dx-2, j, k)];
            }
            
            if (is_last[0]) {
                data[index(dx-1, j, k)] = data[index(1, j, k)];
            }
        }

        //ось Y  
#pragma omp for nowait
        for (int p = 0; p < dx * dz; p++) {
            int i = p % dx;
            int k = (p / dx) % dz;
            
            if (is_first[1]) {
                data[index(i, 0, k)] = data[index(i, dy-2, k)];
            }
            
            if (is_last[1]) {
                data[index(i, dy-1, k)] = data[index(i, 1, k)];
            }
        }

        //ось Z
#pragma omp for
        for (int p = 0; p < dx * dy; p++) {
            int i = p % dx;
            int j = (p / dx) % dy;
            
            if (is_first[2]) {
                data[index(i, j, 0)] = data[index(i, j, dz-2)];
            }
            
            if (is_last[2]) {
                data[index(i, j, dz-1)] = data[index(i, j, 1)];
            }
        }
    }
}


#endif
