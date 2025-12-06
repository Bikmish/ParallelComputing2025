build: 
	mkdir -p stats
	clang++ -O3 -std=c++11 -fopenmp solver.cpp main.cpp -o solver

build_polus: 
	mkdir -p stats
	g++ -O3 -std=c++11 -fopenmp solver.cpp main.cpp -o solver

test_n256_l1_thAll: build
	for threads in 1 2 4 8 16 32 ; do \
		./solver 256 1 $$threads; \
	done \

test_n256_lPi_thAll: build
	for threads in 1 2 4 8 16 32 ; do \
		./solver 256 Pi $$threads; \
	done \

test_n256_l1_thAll_polus: build_polus
	for threads in 1 2 4 8 16 32 ; do \
		bsub -n 1 -q short -W 00:30 -J omp_job\_256\_$$threads\_1 -o stats\_$$N\_$$threads\_$$type.out -e stats\_256\_$$threads\_$$type.err -R "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]"  ./solver 256 1 $$threads; \
	done \

test_full_config: build
	for N in 128 256 512 ; do \
		for type in '1' 'Pi' ; do \
		  	for threads in 1 2 4 8 16 32 ; do \
				./solver $$N $$type $$threads; \
			done \
		done \
	done \

test_full_config_polus: build_polus
	for N in 128 256 512 ; do \
		for type in '1' 'Pi' ; do \
		  	for threads in 1 2 4 8 16 32 ; do \
				bsub -n 1 -q short -W 00:30 -J omp_job\_$$N\_$$threads\_$$type -o stats\_$$N\_$$threads\_$$type.out -e stats\_$$N\_$$threads\_$$type.err -R "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]"  ./solver $$N $$type $$threads; \
			done \
		done \
	done \