#ifndef GRID_H
#define GRID_H

class Grid {
  public:
    int N;
    double Lx, Ly, Lz, h_x, h_y, h_z;

    Grid(int N, double Lx, double Ly, double Lz) {
        this->N = N;
        this->Lx = Lx;
        this->Ly = Ly;
        this->Lz = Lz;
        this->h_x = Lx / N;
        this->h_y = Ly / N;
        this->h_z = Lz / N;
    }

    inline int getLinear(const int& i, const int& j, const int& k) const {
        return (i * (this->N + 1) + j) * (this->N + 1) + k;
    }
};
#endif
