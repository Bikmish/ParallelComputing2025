MPICXX = mpixlC
CXXFLAGS = -O3 -qsmp=omp -std=c++11
TARGET = solverMPI

N_VALUES = 128 256 512
L_VALUES = 1 Pi
MPI_PROCS = 4 8
OMP_THREADS = 1 2 4 8
L_VALUES_MAX = 1 Pi
MPI_PROCS_MAX = 8
OMP_THREADS_MAX = 20

STATS_DIR_MAX = stats_mpi_omp_max
STATS_DIR = stats_mpi_omp

all: build

build:
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) main.cpp
	@echo "Program $(TARGET) compiled successfully"

$(STATS_DIR):
	mkdir -p $(STATS_DIR)

$(STATS_DIR_MAX):
	mkdir -p $(STATS_DIR_MAX)

run_all: build $(STATS_DIR)
	@echo "Starting all MPI+OpenMP configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				for t in $(OMP_THREADS); do \
					echo "Submitting job: N=$$n, L=$$l, MPI=$$p, OMP=$$t"; \
					bsub -n $$p \
					     -o "$(STATS_DIR)/out_$${n}_$${l}_$${p}_$${t}.%J.out" \
					     -e "$(STATS_DIR)/err_$${n}_$${l}_$${p}_$${t}.%J.err" \
					     -R "affinity[core(1)]" \
					     -m "polus-c3-ib polus-c4-ib" \
					     -J "mpi_omp_$${n}_$${l}_$${p}_$${t}" \
					     "OMP_NUM_THREADS=$$t mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
					sleep 1; \
				done \
			done \
		done \
	done
	@echo "All jobs submitted. Check $(STATS_DIR) for results."

MPI_PROCS_1 = 1 4 8 16 32
run_all_1: build $(STATS_DIR)
	@echo "Starting all MPI configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS_1); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p, OMP=1"; \
				bsub -n $$p \
					    -o "$(STATS_DIR)/out_$${n}_$${l}_$${p}_{1}.%J.out" \
					    -e "$(STATS_DIR)/err_$${n}_$${l}_$${p}_{1}.%J.err" \
					    -R "affinity[core(1)]" \
					    -m "polus-c3-ib polus-c4-ib" \
					    -J "mpi_omp_$${n}_$${l}_$${p}_{1}" \
					    "OMP_NUM_THREADS=1 mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 1; \
			done \
		done \
	done
	@echo "All jobs submitted. Check $(STATS_DIR) for results."

run_all_max: build $(STATS_DIR_MAX)
	@echo "Starting maximum configuration (160 threads)..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES_MAX); do \
			echo "Submitting job: N=$$n, L=$$l, MPI=$(MPI_PROCS_MAX), OMP=$(OMP_THREADS_MAX)"; \
			bsub -n $(MPI_PROCS_MAX) \
			     -o "$(STATS_DIR_MAX)/out_$${n}_$${l}_$(MPI_PROCS_MAX)_$(OMP_THREADS_MAX).%J.out" \
			     -e "$(STATS_DIR_MAX)/err_$${n}_$${l}_$(MPI_PROCS_MAX)_$(OMP_THREADS_MAX).%J.err" \
			     -R "affinity[core(1)]" \
			     -m "polus-c3-ib polus-c4-ib" \
			     -J "mpi_omp_max_$${n}_$${l}" \
			     "OMP_NUM_THREADS=$(OMP_THREADS_MAX) mpiexec -n $(MPI_PROCS_MAX) ./$(TARGET) $$n $$l $(MPI_PROCS_MAX)"; \
			sleep 1; \
		done \
	done
	@echo "Maximum configuration jobs submitted. Check $(STATS_DIR_MAX) for results."

clean:
	rm -f $(TARGET)
	rm -rf $(STATS_DIR)

help:
	@echo "Available targets:"
	@echo "  build      - Compile the MPI+OpenMP program"
	@echo "  run_all    - Run all configurations for Table 4"
	@echo "  clean      - Remove compiled binary and results"
	@echo "  help       - Show this help message"

.PHONY: all build run_all clean help