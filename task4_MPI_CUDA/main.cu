#include <cstdio>
#include <iostream>
#include <cmath>
#include <memory>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "solver.h"
#include "mpi_utils.h"

TimersArray timers;

double hx, hy, hz, Lx, Ly, Lz;
int dx, dy, dz;
solver_params params;

int main(int argc, char **argv)
{
    int N, num_procs, rank = -1;

    // инициализируем MPI и определяем ранг процесса
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    timers.total.start();
    timers.init.start();

    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " expects either 3 or 5 parameters:\n"
                  << "\t- 3 parameters: N L_val num_procs - grid size & general bound for dimensions (Lx=Ly=Lz=L_val) & number of procs\n"
                  << "\t- 5 parameters: N Lx Ly Lz num_procs - grid size & custom values for every dimension & number of procs" << std::endl;
        return 1;
    }

    N = std::stoi(argv[1]);
    if (argc == 4) {
        double L_val = (std::string(argv[2]) == "Pi") ? M_PI : std::stod(argv[2]);
        num_procs = std::stoi(argv[3]);
        Lx = Ly = Lz = L_val;
    } else { //argc == 6
        Lx = (std::string(argv[2]) == "Pi") ? M_PI : std::stod(argv[2]);
        Ly = (std::string(argv[3]) == "Pi") ? M_PI : std::stod(argv[3]);
        Lz = (std::string(argv[4]) == "Pi") ? M_PI : std::stod(argv[4]);
        num_procs = std::stoi(argv[5]);
    }

    if (N <= 0 || Lx <= 0 || Ly <= 0 || Lz <= 0 || num_procs <= 0) {
        std::cerr << "Error: N, Lx, Ly, Lz must be positive and num_procs must be non-negative!" << std::endl;
        return 1;
    }

    hx = Lx / (N - 1);
    hy = Ly / (N - 1); 
    hz = Lz / (N - 1);

    // число процессов по каждой из оси решетки
    int dims[num_dims];
    balance_process_grid(dims, num_dims, num_procs);

    // решетка является периодической (для установки граничных условий)
    int periods[num_dims];
    for (int d = 0; d < num_dims; d++)
        periods[d] = 1;

    // число узлов решетки для процесса по каждой из осей
    int nodes[num_dims];
    for (int d = 0; d < num_dims; d++) {
        nodes[d] = (int) ceil(N / (double) dims[d]);
        if (!nodes[d]) {
            std::cerr << "[ERROR] Invalid grid split" << std::endl;
            return 1;
        }
    }

    // равномерно распределяем процессы между GPU
    int num_cuda_devices;
    cudaGetDeviceCount(&num_cuda_devices);
    cudaSetDevice(rank % num_cuda_devices);

    // вывод информации о разбиении
    if (!rank) {
        std::cout << N << ' ' << STEPS << ' ' << num_procs << std::endl;
        for (int d = 0; d < num_dims; d++) {
            std::cout << "axis" << d << '\t'
                      << dims[d] << '\t' << nodes[d] << std::endl;
        }
        std::cout << "Number of cuda devices: " << num_cuda_devices << std::endl;
    }

    int curr_cuda_device;
    cudaGetDevice(&curr_cuda_device);
    std::cout << rank << '->' << curr_cuda_device << ' ' << std::endl;

    // создаем топологию
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, num_dims, dims, periods, 0, &comm_cart);

    // координаты процесса в системе декартовой решетки
    int coords[num_dims];
    MPI_Cart_coords(comm_cart, rank, num_dims, coords);

    // вычисляем соседей для процесса по каждой из осей
    int rank_prev[num_dims], rank_next[num_dims];
    for (int d = 0; d < num_dims; d++) {
        MPI_Cart_shift(comm_cart, d, +1, &rank_prev[d], &rank_next[d]);
    }

    // индикаторы того, что процесс является первым и/или последним по каждой из осей
    bool is_first[num_dims], is_last[num_dims];
    for (int d = 0; d < num_dims; d++) {
        is_first[d] = (!coords[d]);
        is_last[d] = (coords[d] == dims[d] - 1);
    }

    // минимальные и максимальные рабочие индексы
    const int i_min = coords[0] * nodes[0], i_max = std::min(N, (coords[0] + 1) * nodes[0]) - 1;
    const int j_min = coords[1] * nodes[1], j_max = std::min(N, (coords[1] + 1) * nodes[1]) - 1;
    const int k_min = coords[2] * nodes[2], k_max = std::min(N, (coords[2] + 1) * nodes[2]) - 1;

    // ширина области в индексах
    // храним еще и обменные области (по 2е на каждую ось), помимо рабочих областей
    dx = i_max - i_min + 1 + 2;
    dy = j_max - j_min + 1 + 2;
    dz = k_max - k_min + 1 + 2;

    params = {dx, dy, dz,
              hx, hy, hz,
              Lx, Ly, Lz,
              TAU,
              i_min, j_min, k_min,
              pack_boundary_mask(is_first, is_last)};
    
    // подсчет ошибки
    EstimateError error_cumm, error_curr, error_proc;

    cudaError_t err;

    // выделяем память на GPU
    double *u_data[TIME_LAYERS_WINDOW_SIZE], *u_error;
    for (int p = 0; p < TIME_LAYERS_WINDOW_SIZE; p++)
        cudaMalloc((void **) &u_data[p], sizeof(double) * dx * dy * dz);
    cudaMalloc((void **) &u_error, sizeof(double) * (dx - 2) * (dy - 2) * (dz - 2));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Memory GPU allocation failed.\n");
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    timers.init.pause();

    // засекаем время
    MPITimer timer;
    timer.start();

    // определяем разбиение на GPU (обменные области заполняются, но не вычисляются)
    dim3 threads(8, 8, 8);
    dim3 internalBlocks(split(dx - 2, threads.x), split(dy - 2, threads.y), split(dz - 2, threads.z));
    dim3 initBlocks(split(dx, threads.x), split(dy, threads.y), split(dz, threads.z));

    dim3 blocks;
    // 0-й, 1-й и последующий временные слои: u^0_ijk, u^1_ijk, ..
    for (int n = 0; n < STEPS; n++) {
        blocks = n == 0 ? initBlocks : internalBlocks;
        solver_step <<< blocks, threads >>> (
            u_data[n % TIME_LAYERS_WINDOW_SIZE], 
            n > 0 ? u_data[(n - 1) % TIME_LAYERS_WINDOW_SIZE] : 0, 
            n > 1 ? u_data[(n - 2) % TIME_LAYERS_WINDOW_SIZE] : 0, 
            n, params);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Fail during solver's step on iteration: %d\n", n);
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        if (n > 0)
        {
            // обмены граничными областями между процессами по оси X
            send_recv_forward_x (u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);
            send_recv_backward_x(u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);

            // обмены граничными областями между процессами по оси Y
            send_recv_forward_y (u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[1], rank_next[1], is_first[1], is_last[1]);
            send_recv_backward_y(u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[1], rank_next[1], is_first[1], is_last[1]);

            // обмены граничными областями между процессами по оси Z
            send_recv_forward_z (u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[2], rank_next[2], is_first[2], is_last[2]);
            send_recv_backward_z(u_data[n % TIME_LAYERS_WINDOW_SIZE], comm_cart, rank_prev[2], rank_next[2], is_first[2], is_last[2]);

            cudaDeviceSynchronize();

            {
                CudaScopeTimerCallback cb(&timers.copy);

                set_boundary_conditions(u_data[n % TIME_LAYERS_WINDOW_SIZE], is_first, is_last, params);
            }
        }

        cudaDeviceSynchronize();

        TimerScopePauseCallback callback(timer);

        error_curr.mse = 0;
        error_curr.max = 0;

        compute_mse_error <<< blocks, threads >>> (u_error, u_data[n % TIME_LAYERS_WINDOW_SIZE], params, n);
        cudaDeviceSynchronize();
        error_proc.mse = thrust::reduce(
            thrust::device,
            u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
            0.0, thrust::plus<double>()
        );

        compute_max_error <<< blocks, threads >>> (u_error, u_data[n % TIME_LAYERS_WINDOW_SIZE], params, n);
        cudaDeviceSynchronize();
        error_proc.max = thrust::reduce(
            thrust::device,
            u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
            0.0, thrust::maximum<double>()
        );

        MPI_Reduce(&error_proc.mse, &error_curr.mse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
        MPI_Reduce(&error_proc.max, &error_curr.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

        if (!rank) {
            error_curr.mse /= pow(N, 3);
            error_cumm.mse += error_curr.mse;

            if (error_curr.max > error_cumm.max)
                error_cumm.max = error_curr.max;
        }

        if (!rank) {
            printf("[iter %03d]", n);
            printf(" RMSE = %.15f; MAX = %.15f;", sqrt(error_curr.mse), error_curr.max);
            printf(" Time = %.10f sec.\n", timer.delta());
        }
    }

    timer.pause();

    if (!rank) {
        printf("Final RMSE = %.15f; MAX = %.15f\n", sqrt(error_cumm.mse / STEPS), error_cumm.max);
        printf("Task elapsed in: %.10f sec.\n", timer.delta());
    }

    timers.free.start();

    // освобождаем память
    for (int p = 0; p < TIME_LAYERS_WINDOW_SIZE; p++)
        cudaFree(u_data[p]);
    cudaFree(u_error);

    timers.free.pause();
    timers.total.pause();

    MPI_Finalize();

    if (!rank) {
        printf("\n");
        printf("Time total:     %.10f\n", timers.total.delta());
        printf("Time init:      %.10f\n", timers.init.delta());
        printf("Time logic:     %.10f\n", timer.delta());
        printf("Time sendrecv:  %.10f\n", timers.sendrecv.delta());
        printf("Time copy:      %.10f\n", timers.copy);
        printf("Time free:      %.10f\n", timers.free.delta());
    }

    return 0;
}
