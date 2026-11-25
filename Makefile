CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -mcpu=native -Wall

bench: bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

# One core only
run1: bench
	OMP_NUM_THREADS=1 ./bench

run: bench
	./bench

asm:
	$(CXX) $(CXXFLAGS) -S bench.cpp -o bench.s   # inspect fmla v*.4s in the microkernel

clean:
	rm -f bench bench.s