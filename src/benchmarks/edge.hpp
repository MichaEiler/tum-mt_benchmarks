#ifndef __BENCH_BENCHMARKS_EDGE_HPP
#define __BENCH_BENCHMARKS_EDGE_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * This benchmark implements a very simple version of edge detection on images.
 * Its main use is to show the effect on performance of dynamic branching.
 */
class Edge : public BenchmarkBase {
private:

	std::shared_ptr<cl::Buffer> _imageBuffer = nullptr;
	std::shared_ptr<cl::Buffer> _edgeSteepnessBuffer = nullptr;
	std::shared_ptr<cl::Kernel> _edgeKernel = nullptr;
	std::shared_ptr<cl::Kernel> _optimizedEdgeKernel = nullptr;
	std::shared_ptr<cl::Program> _program = nullptr;

	/**
	 * Compiles the kernel, initialize buffers and sets the kernel arguments.
	 */
	int InitContext();

	/**
	 * Generates an input image.
	 *
	 * If USE_STATIC_IMAGE is set to true it is simply filled with one color,
	 * otherwise (default) the image data is random.
	 * Use this flag to analyze dynamic branchbehavior.
	 */
	void InitData();

	/**
	 * Execute the kernel multiple times and generate statistics.
	 */
	void ExecuteKernel();

	/**
	 * Release all resoruces such as buffers, kernels and the program instance.
	 */
	void Cleanup();

public:
    explicit Edge(std::shared_ptr<ComputeController> controller);

    virtual ~Edge();

    /**
     * Execute the benchmark.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_EDGE_HPP
