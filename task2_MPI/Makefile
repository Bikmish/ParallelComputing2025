MPICXX = mpixlC
CXXFLAGS = -O3 -std=c++11
TARGET = solverMPI_only

N_VALUES = 128 256 512
L_VALUES = 1 Pi
MPI_PROCS = 1 2 4 8 16 32

STATS_DIR = stats_mpi_only

all: build

build:
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) main.cpp
	@echo "Program $(TARGET) compiled successfully"

$(STATS_DIR):
	mkdir -p $(STATS_DIR)

run_all: build $(STATS_DIR)
	@echo "Starting all MPI configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p"; \
				bsub -n $$p \
				     -o "$(STATS_DIR)/out_$${n}_$${l}_$${p}.%J.out" \
				     -e "$(STATS_DIR)/err_$${n}_$${l}_$${p}.%J.err" \
				     -R "affinity[core(1)]" \
				     -m "polus-c3-ib polus-c4-ib" \
				     -J "mpi_only_$${n}_$${l}_$${p}" \
				     "mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 1; \
			done \
		done \
	done
	@echo "All MPI jobs submitted. Check $(STATS_DIR) for results."

clean:
	rm -f $(TARGET)
	rm -rf $(STATS_DIR)

help:
	@echo "Available targets:"
	@echo "  build      - Compile the pure MPI program"
	@echo "  run_all    - Run all configurations for MPI only"
	@echo "  clean      - Remove compiled binary and results"
	@echo "  help       - Show this help message"

.PHONY: all build run_all clean help