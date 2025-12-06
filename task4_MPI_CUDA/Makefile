ARCH ?= sm_35
HOST_COMP ?= mpic++
TARGET = solverCUDA
SOURCE = main.cu
CFLAGS = -x cu -Xcompiler -Wall -std=c++11
CLINKFLAGS = -O3 -arch=$(ARCH)
NVCC = nvcc

$(TARGET): $(SOURCE)
	$(NVCC) $(CLINKFLAGS) -ccbin $(HOST_COMP) -o $@ $< $(CFLAGS)

clean:
	rm -f $(TARGET)

N_VALUES = 128 256 512 1024
L_VALUES = Pi
MPI_PROCS = 1 2 4 8 16

STATS_DIR_1GPU = stats_cuda_1gpu
STATS_DIR_1GPU_EX = stats_cuda_1gpu_ex
STATS_DIR_2GPU = stats_cuda_2gpu
STATS_DIR_2GPU_EX = stats_cuda_2gpu_ex
TEST_DIR = tests

$(STATS_DIR_1GPU):
	mkdir -p $(STATS_DIR_1GPU)

$(STATS_DIR_1GPU_EX) $(STATS_DIR_2GPU) $(STATS_DIR_2GPU_EX):
	mkdir -p $(STATS_DIR_1GPU_EX) $(STATS_DIR_2GPU) $(STATS_DIR_2GPU_EX)

$(TEST_DIR):
	mkdir -p $(TEST_DIR)

run_test: build $(TEST_DIR)
	@echo "Submitting test job..."
	bsub -n 8 \
	     -o "$(TEST_DIR)/out_cuda_test_256_Pi_8.%J.out" \
	     -e "$(TEST_DIR)/err_cuda_test_256_Pi_8.%J.err" \
	     -R "affinity[core(1)]" \
	     -R "span[hosts=1]" \
	     -m "polus-c3-ib polus-c4-ib" \
	     -gpu "num=1:mode=shared" \
	     -J "cuda_test_256_Pi_8" \
	     "mpiexec -n 8 ./$(TARGET) 256 Pi 8"

run_all_single_gpu: build $(STATS_DIR_1GPU)
	@echo "Starting all single GPU configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p (1 GPU)"; \
				bsub -n $$p \
				     -o "$(STATS_DIR_1GPU)/out_$${n}_$${l}_$${p}_1gpu.%J.out" \
				     -e "$(STATS_DIR_1GPU)/err_$${n}_$${l}_$${p}_1gpu.%J.err" \
				     -R "affinity[core(1)]" \
				     -R "span[hosts=1]" \
				     -m "polus-c3-ib polus-c4-ib" \
				     -gpu "num=1:mode=shared" \
				     -J "cuda_1gpu_$${n}_$${l}_$${p}" \
				     "mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 2; \
			done \
		done \
	done
	@echo "All single GPU jobs submitted. Check $(STATS_DIR_1GPU) for results."

run_all_single_gpu_ex: build $(STATS_DIR_1GPU_EX)
	@echo "Starting all single GPU exclusive configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p (1 GPU exclusive)"; \
				bsub -n $$p \
				     -o "$(STATS_DIR_1GPU_EX)/out_$${n}_$${l}_$${p}_1gpu_ex.%J.out" \
				     -e "$(STATS_DIR_1GPU_EX)/err_$${n}_$${l}_$${p}_1gpu_ex.%J.err" \
				     -R "affinity[core(1)]" \
				     -R "span[hosts=1]" \
				     -m "polus-c3-ib polus-c4-ib" \
				     -gpu "num=1:mode=exclusive_process" \
				     -J "cuda_1gpu_ex_$${n}_$${l}_$${p}" \
				     "mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 2; \
			done \
		done \
	done
	@echo "All single GPU exclusive jobs submitted. Check $(STATS_DIR_1GPU_EX) for results."

run_all_double_gpu: build $(STATS_DIR_2GPU)
	@echo "Starting all double GPU shared configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p (2 GPU shared)"; \
				bsub -n $$p \
				     -o "$(STATS_DIR_2GPU)/out_$${n}_$${l}_$${p}_2gpu.%J.out" \
				     -e "$(STATS_DIR_2GPU)/err_$${n}_$${l}_$${p}_2gpu.%J.err" \
				     -R "affinity[core(1)]" \
				     -R "span[hosts=1]" \
				     -m "polus-c3-ib polus-c4-ib" \
				     -gpu "num=2:mode=shared" \
				     -J "cuda_2gpu_$${n}_$${l}_$${p}" \
				     "mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 2; \
			done \
		done \
	done
	@echo "All double GPU shared jobs submitted. Check $(STATS_DIR_2GPU) for results."

run_all_double_gpu_ex: build $(STATS_DIR_2GPU_EX)
	@echo "Starting all double GPU exclusive configurations..."
	@for n in $(N_VALUES); do \
		for l in $(L_VALUES); do \
			for p in $(MPI_PROCS); do \
				echo "Submitting job: N=$$n, L=$$l, MPI=$$p (2 GPU exclusive)"; \
				bsub -n $$p \
				     -o "$(STATS_DIR_2GPU_EX)/out_$${n}_$${l}_$${p}_2gpu_ex.%J.out" \
				     -e "$(STATS_DIR_2GPU_EX)/err_$${n}_$${l}_$${p}_2gpu_ex.%J.err" \
				     -R "affinity[core(1)]" \
				     -R "span[hosts=1]" \
				     -m "polus-c3-ib polus-c4-ib" \
				     -gpu "num=2:mode=exclusive_process" \
				     -J "cuda_2gpu_ex_$${n}_$${l}_$${p}" \
				     "mpiexec -n $$p ./$(TARGET) $$n $$l $$p"; \
				sleep 2; \
			done \
		done \
	done
	@echo "All double GPU exclusive jobs submitted. Check $(STATS_DIR_2GPU_EX) for results."

build: $(TARGET)

.PHONY: clean build run_all_single_gpu run_test