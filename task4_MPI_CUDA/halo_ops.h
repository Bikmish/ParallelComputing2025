#include "utils.h"

__global__ void update_halo_x(double *data, solver_params params, int i_dst, int i_src)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    data[index(i_dst, j, k, params)] = data[index(i_src, j, k, params)];
}

__global__ void update_halo_y(double *data, solver_params params, int j_dst, int j_src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    data[index(i, j_dst, k, params)] = data[index(i, j_src, k, params)];
}

__global__ void update_halo_z(double *data, solver_params params, int k_dst, int k_src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    data[index(i, j, k_dst, params)] = data[index(i, j, k_src, params)];
}

__global__ void pack_slice_to_halo_buffer_x(double *buffer, double *data, solver_params params, int i)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    int p = j + k * params.dy;
    buffer[p] = data[index(i, j, k, params)];
}

__global__ void unpack_slice_from_halo_buffer_x(double *buffer, double *data, solver_params params, int i)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    data[index(i, j, k, params)] = buffer[j + k * params.dy];
}

__global__ void pack_slice_to_halo_buffer_y(double *buffer, double *data, solver_params params, int j)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    buffer[i + k * params.dx] = data[index(i, j, k, params)];
}

__global__ void unpack_slice_to_halo_buffer_y(double *buffer, double *data, solver_params params, int j)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    data[index(i, j, k, params)] = buffer[i + k * params.dx];
}

__global__ void pack_slice_to_halo_buffer_z(double *buffer, double *data, solver_params params, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    buffer[i + j * params.dx] = data[index(i, j, k, params)];
}

__global__ void unpack_slice_to_halo_buffer_z(double *buffer, double *data, solver_params params, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    data[index(i, j, k, params)] = buffer[i + j * params.dx];
}