#ifndef SOLVER_H
#define SOLVER_H

#include "utils.h"
#include <cmath>
#include "halo_ops.h"

#define TAU 0.00001
#define STEPS 20
#define TIME_LAYERS_WINDOW_SIZE 3

__host__ __device__ double analytical_solution(double x, double y, double z, double t, solver_params params) {
    double at = M_PI * sqrt(4.0/(params.Lx*params.Lx) + 16.0/(params.Ly*params.Ly) + 36.0/(params.Lz*params.Lz));
    return sin(2 * M_PI * x / params.Lx) * 
           sin(4 * M_PI * y / params.Ly) * 
           sin(6 * M_PI * z / params.Lz) *
           cos(at * t);
}

__device__ double calculate_second_derivative(double prev_val, double curr_val, double next_val, double step_size)
{
    return (prev_val - 2.0 * curr_val + next_val) / (step_size * step_size);
}

__device__ double laplace(const double data[], int i, int j, int k, solver_params params)
{
    const int curr = index(i, j, k, params);
    const double curr_val = data[curr];
    double val = 0.0;

    //2-я производная по X
    val += calculate_second_derivative(data[index(i-1, j, k, params)], curr_val, data[index(i+1, j, k, params)], params.hx);

    //2-я производная по Y
    val += calculate_second_derivative(data[index(i, j-1, k, params)], curr_val, data[index(i, j+1, k, params)], params.hy);

    //2-я производная по Z
    val += calculate_second_derivative(data[index(i, j, k-1, params)], curr_val, data[index(i, j, k+1, params)], params.hz);

    return val;
}

__global__ void compute_mse_error(double *err, const double *data, solver_params params, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    // пропускаем правую обменную область и выход за границы
    if (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1)
        return;

    int p = (i - 1) + (j - 1) * (params.dx - 2) + (k - 1) * (params.dx - 2) * (params.dy - 2);
    double u_analytical = analytical_solution((params.i_min + i - 1) * params.hx, (params.j_min + j - 1) * params.hy, (params.k_min + k - 1) * params.hz, n * params.tau, params);
    err[p] = pow(data[index(i, j, k, params)] - u_analytical, 2.0);
}

__global__ void compute_max_error(double *err, const double *data, solver_params params, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    // пропускаем правую обменную область и выход за границы
    if (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1)
        return;

    int p = (i - 1) + (j - 1) * (params.dx - 2) + (k - 1) * (params.dx - 2) * (params.dy - 2);
    double u_analytical = analytical_solution((params.i_min + i - 1) * params.hx, (params.j_min + j - 1) * params.hy, (params.k_min + k - 1) * params.hz, n * params.tau, params);
    double value = data[index(i, j, k, params)] - u_analytical;
    err[p] = (value < 0) ? -value : value;
}

void set_boundary_conditions(double *data, bool is_first_arr[3], bool is_last_arr[3], solver_params params)
{
    //ось X
    if (is_first_arr[0] || is_last_arr[0])
    {
        if (is_first_arr[0])
        {
            dim3 threads(1, 16, 16);
            dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));
            update_halo_x <<< blocks, threads >>> (data, params, 1, 0);
        }
    }


    //оси Y
    if (is_first_arr[1] || is_last_arr[1])
    {
        if (is_first_arr[1])
        {
            dim3 threads(16, 1, 16);
            dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));
            update_halo_y <<< blocks, threads >>> (data, params, 1, 0);
        }
    }

    //ось Z
    if (is_first_arr[2] || is_last_arr[2])
    {
        if (is_first_arr[2])
        {
            dim3 threads(16, 16, 1);
            dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);
            update_halo_z <<< blocks, threads >>> (data, params, 1, 0);
        }
    }
}

__global__ void solver_step(double *p_next, double *p_curr, double *p_prev, int n, solver_params params)
{
    int iter_shift = (int)(n > 0);
    int i = blockDim.x * blockIdx.x + threadIdx.x + iter_shift;
    int j = blockDim.y * blockIdx.y + threadIdx.y + iter_shift;
    int k = blockDim.z * blockIdx.z + threadIdx.z + iter_shift;

    bool is_first[num_dims], is_last[num_dims];
    unpack_boundary_mask(is_first, is_last, params.fl_mask);

    // пропускаем граничные, обменные области и области, выходящие за границы
    if (n > 0 && (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1 || is_first[0] && i == 1 || is_first[1] && j == 1 || is_first[2] && k == 1) ||
        n == 0 && (i >= params.dx || j >= params.dy || k >= params.dz))
        return;

    // считаем 0-й, 1-й и последующие временные слои u^0_ijk, u^1_ijk, ...,u^0_ijk,... по ф-лам 10, 12 и (*) соответственно
    // *ниже (N) означает, что используется N-я формула из условия задания, (*) - формула из п.3 условия (почему-то без номера)
    //(10): u^0_ijk = phi(xi, yj, zk) = {из (3)} = u|t=0
    //(12): u^1_ijk = u^0_ijk + a^2*(a^2 * TAU^2 / 2) * laplace(phi(xi, yj, zk)) = {a^2 = 1 + из (3) phi=u|t=0=u0} = u^0_ijk + (a^2 * TAU^2 / 2) * laplace(u0)
    //(*) (u^{n+1}_ijk - 2u^n_ijk + u^{n-1}_ijk) / TAU^2 = a^2 * laplace(u^n), учтём a^2=1 => => u^{n+1}_ijk = 2u^n_ijk - u^{n-1}_ijk + TAU^2 * laplace(u^n)
    int p = index(i, j, k, params);
    p_next[p] = n == 0 ? 
                    analytical_solution((params.i_min + i - 1) * params.hx, (params.j_min + j - 1) * params.hy, (params.k_min + k - 1) * params.hz, 0, params) : (
                        n == 1 ?
                            p_curr[p] + 0.5 * params.tau * params.tau * laplace(p_curr, i, j, k, params) :
                            2 * p_curr[p] - p_prev[p] + params.tau * params.tau * laplace(p_curr, i, j, k, params));
}


#endif //SOLVER_H
