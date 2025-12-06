#include <memory>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>

#include "solver.h"
#include "mpi_utils.h"


TimersArray timers;

double hx, hy, hz, Lx, Ly, Lz;
int dx, dy, dz;

int main(int argc, char **argv)
{
    timers.total.start();
    timers.init.start();

    int N, num_procs, rank = -1;

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

    // размерность декартовой решетки
    int ndim = 3;

    // число процессов по каждой из оси решетки
    int dims[ndim];
    balance_process_grid(dims, ndim, num_procs);

    // решетка является периодической (для установки граничных условий)
    int periods[ndim];
    for (int d = 0; d < ndim; d++)
        periods[d] = 1;

    // число узлов решетки для процесса по каждой из осей
    int nodes[ndim];
    for (int d = 0; d < ndim; d++) {
        nodes[d] = (int) ceil(N / (double) dims[d]);
        if (!nodes[d]) {
            std::cerr << "[ERROR] Invalid grid split" << std::endl;
            return 1;
        }
    }

    // инициализируем MPI и определяем ранг процесса
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc == 4) {
            std::cout << "Using uniform dimensions: Lx = Ly = Lz = " << Lx << " with " << num_procs << " procs" << std::endl;
        } else {
            std::cout << "Using custom dimensions: Lx = " << Lx 
                      << ", Ly = " << Ly 
                      << ", Lz = " << Lz << " with " << num_procs << " procs" << std::endl;
        }
    }

    // вывод информации о разбиении
    if (!rank) {
        std::cout << N << ' ' << STEPS << ' ' << num_procs << std::endl;
        for (int d = 0; d < ndim; d++) {
            std::cout << "axis" << d << '\t'
                      << dims[d] << '\t' << nodes[d] << std::endl;
        }
    }

    // создаем топологию
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, 0, &comm_cart);

    // координаты процесса в системе декартовой решетки
    int coords[ndim];
    MPI_Cart_coords(comm_cart, rank, ndim, coords);

    // вычисляем соседей для процесса по каждой из осей
    int rank_prev[ndim], rank_next[ndim];
    for (int d = 0; d < ndim; d++) {
        MPI_Cart_shift(comm_cart, d, +1, &rank_prev[d], &rank_next[d]);
    }

    // индикаторы того, что процесс является первым и/или последним по каждой из осей
    bool is_first[ndim], is_last[ndim];
    for (int d = 0; d < ndim; d++) {
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

    // подсчет ошибки
    ErrorInfo error_cumm, error_curr, error_proc;

    // выделяем память
    double* u_data[TIME_LAYERS_WINDOW_SIZE];
    for (int p = 0; p < TIME_LAYERS_WINDOW_SIZE; p++)
        u_data[p] = new double[dx * dy * dz];

    timers.init.pause();

    // засекаем время
    MPITimer timer;
    timer.start();

    // 0-й, 1-й и последующий временные слои: u^0_ijk, u^1_ijk, ...
    for (int n = 0; n < STEPS; n++) {
        for (int p = 0; p < dx * dy * dz; p++) {
            int i = p % dx;
            int j = (p / dx) % dy;
            int k = (p / dx / dy) % dz;

            // пропускаем граничные и обменные области
            if (i == 0 || j == 0 || k == 0 || i == dx - 1 || j == dy - 1 || k == dz - 1 ||
                (is_first[0] && i == 1) || (is_last[0] && i == dx - 2) ||
                (is_first[1] && j == 1) || (is_last[1] && j == dy - 2) ||
                (is_first[2] && k == 1) || (is_last[2] && k == dz - 2))
                continue;

            // считаем 0-й, 1-й и последующие временные слои u^0_ijk, u^1_ijk, ...,u^0_ijk,... по ф-лам 10, 12 и (*) соответственно
            // *ниже (N) означает, что используется N-я формула из условия задания, (*) - формула из п.3 условия (почему-то без номера)
            //(10): u^0_ijk = phi(xi, yj, zk) = {из (3)} = u|t=0
            //(12): u^1_ijk = u^0_ijk + a^2*(a^2 * TAU^2 / 2) * laplace(phi(xi, yj, zk)) = {a^2 = 1 + из (3) phi=u|t=0=u0} = u^0_ijk + (a^2 * TAU^2 / 2) * laplace(u0)
            //(*) (u^{n+1}_ijk - 2u^n_ijk + u^{n-1}_ijk) / TAU^2 = a^2 * laplace(u^n), учтём a^2=1 => => u^{n+1}_ijk = 2u^n_ijk - u^{n-1}_ijk + TAU^2 * laplace(u^n)
            u_data[n % TIME_LAYERS_WINDOW_SIZE][p] = n == 0 ?
                                    analytical_solution((i_min + i - 1) * hx, (j_min + j - 1) * hy, (k_min + k - 1) * hz, 0.0, Lx, Ly, Lz) : (
                                    n == 1 ? 
                                        u_data[n-1][p] + 0.5 * TAU * TAU * (laplace(u_data[n-1], i, j, k)) :
                                        2 * u_data[(n - 1) % TIME_LAYERS_WINDOW_SIZE][p] - u_data[(n - 2) % TIME_LAYERS_WINDOW_SIZE][p] + TAU * TAU * ( laplace(u_data[(n - 1) % TIME_LAYERS_WINDOW_SIZE], i, j, k)));
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

            {
                TimerScopeUnpauseCallback cb(timers.copy);

                set_boundary_conditions(u_data[n % TIME_LAYERS_WINDOW_SIZE], is_first, is_last);
            }
        }

        
        TimerScopePauseCallback callback(timer);

        error_curr.mse = 0;
        error_curr.max = 0;

        compute_error(&error_proc, u_data[n % TIME_LAYERS_WINDOW_SIZE], n, i_min, j_min, k_min, Lx, Ly, Lz);

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
            printf(" RMSE = %.10f; MAX = %.10f;", sqrt(error_curr.mse), error_curr.max);
            printf(" Time = %.6f sec.\n", timer.delta());
        }
    }

    timer.pause();

    if (!rank) {
        printf("Final RMSE = %.10f; MAX = %.10f\n", sqrt(error_cumm.mse / STEPS), error_cumm.max);
        printf("Task elapsed in: %.6f sec.\n", timer.delta());
    }

    timers.free.start();

    // освобождаем память
    for (int p = 0; p < TIME_LAYERS_WINDOW_SIZE; p++)
        delete u_data[p];

    MPI_Finalize();

    timers.free.pause();
    timers.total.pause();

    if (!rank) {
        printf("\n");
        printf("Time total:     %.6f\n", timers.total.delta());
        printf("Time init:      %.6f\n", timers.init.delta());
        printf("Time logic:     %.6f\n", timer.delta());
        printf("Time sendrecv:  %.6f\n", timers.sendrecv.delta());
        printf("Time copy:      %.6f\n", timers.copy.delta());
        printf("Time free:      %.6f\n", timers.free.delta());
    }

    return 0;
}
