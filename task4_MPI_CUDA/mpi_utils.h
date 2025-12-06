#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "utils.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

void balance_process_grid(int* dims, int num_dims_loc, int num_procs)
{
    std::vector<int> divisors;
    std::vector<int> primes;
    std::vector<bool> is_removed(static_cast<int>(std::sqrt(num_procs)) + 1, false);

    for (int i = 2; i < is_removed.size(); i++) {
        if (is_removed[i])
            continue;

        for (int j = i * 2; j < is_removed.size(); j += i) {
            is_removed[j] = true;
        }

        primes.push_back(i);
    }
    
    for (int i = 0; i < primes.size(); i++) {
        int p = primes[i];

        while (num_procs % p == 0) {
            divisors.push_back(p);
            num_procs /= p;
        }

        if (num_procs == 1)
            break;
    }

    if (num_procs != 1)
        divisors.push_back(num_procs);
    
    std::sort(divisors.begin(), divisors.end(), std::greater<int>());
    std::vector<int> split(num_dims_loc, 1);

    //распределяем делители по наименьшему текущему измерению
    for (int i = 0; i < divisors.size(); i++) {
        *std::min_element(split.begin(), split.end()) *= divisors[i];
    }

    std::copy(split.begin(), split.end(), dims);
}

extern TimersArray timers;
extern solver_params params;
void send_recv_forward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_x <<< blocks, threads >>> (data, params, 0, dx - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_x (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_x <<< blocks, threads >>> (dev_buffer, data, params, dx - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_from_halo_buffer_x <<< blocks, threads >>> (dev_buffer, data, params, 0);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_x <<< blocks, threads >>> (data, params, dx - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_x (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_x <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_from_halo_buffer_x <<< blocks, threads >>> (dev_buffer, data, params, dx - 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_forward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_y <<< blocks, threads >>> (data, params, 0, dy - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_y (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_y <<< blocks, threads >>> (dev_buffer, data, params, dy - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_to_halo_buffer_y <<< blocks, threads >>> (dev_buffer, data, params, 0);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_y <<< blocks, threads >>> (data, params, dy - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_y (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_y <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_to_halo_buffer_y <<< blocks, threads >>> (dev_buffer, data, params, dy - 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_forward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_z <<< blocks, threads >>> (data, params, 0, dz - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_z (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_z <<< blocks, threads >>> (dev_buffer, data, params, dz - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_to_halo_buffer_z <<< blocks, threads >>> (dev_buffer, data, params, 0);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        update_halo_z <<< blocks, threads >>> (data, params, dz - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion update_halo_z (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        pack_slice_to_halo_buffer_z <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        unpack_slice_to_halo_buffer_z <<< blocks, threads >>> (dev_buffer, data, params, dz - 1);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

#endif //MPI_UTILS_H
