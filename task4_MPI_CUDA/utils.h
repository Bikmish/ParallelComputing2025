#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <cuda.h>
#include "timer.h"
#define num_dims 3

extern int dx, dy, dz;
extern double hx, hy, hz;

struct TimersArray {
    MPITimer total;
    MPITimer init;
    MPITimer free;
    MPITimer sendrecv;
    double copy;
};

struct EstimateError {
    double mse;
    double max;

    EstimateError() : mse(0), max(0) {}
};

struct solver_params {
    int dx, dy, dz;
    double hx, hy, hz, Lx, Ly, Lz, tau;
    int i_min, j_min, k_min;
    char fl_mask;
};

int split(int N, int num_threads) {
    return ceil(double(N) / num_threads);
}

__device__ int index(int i, int j, int k, solver_params params) {
    return i + j * params.dx + k * params.dx * params.dy;
}

char pack_boundary_mask(const bool *is_first, const bool *is_last)
{
    char fl_mask = 0;
    for (char d = 0; d < num_dims; d++)
        fl_mask |= ( char(is_first[d]) << 1 | char(is_last[d]) ) << (2 * d);
    return fl_mask;
}

__host__ __device__ void unpack_boundary_mask(bool *is_first, bool *is_last, char fl_mask)
{
    for (char d = 0; d < num_dims; d++) {
        is_first[d] = fl_mask & 2;
        is_last[d] = fl_mask & 1;
        fl_mask >>= 2;
    }
}

#endif //UTILS_H
