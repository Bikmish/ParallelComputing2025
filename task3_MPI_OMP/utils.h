#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>
#include <omp.h>


struct ErrorInfo {
    double mse;
    double max;

    ErrorInfo() : mse(0), max(0) {}
};

extern int dx, dy, dz;
inline int index(int i, int j, int k) {
    return i + j * dx + k * dx * dy;
}

#endif
