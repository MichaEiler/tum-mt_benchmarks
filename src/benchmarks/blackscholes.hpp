#ifndef __BENCH_BENCHMARKS_BLACKSCHOLES_HPP
#define __BENCH_BENCHMARKS_BLACKSCHOLES_HPP

#include "../benchmarkbase.hpp"

#include <memory>
#include <vector>

namespace benchmarks {

/**
 * This benchmark executes a kernel simulation parts of financial market.
 * For more information read: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
 */
class BlackScholes : public BenchmarkBase {
private:
    std::shared_ptr<cl::Buffer> _randBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _callPriceBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _putPriceBuffer = nullptr;
    std::shared_ptr<cl::Kernel> _scalarKernel = nullptr;
    std::shared_ptr<cl::Kernel> _vectorizedKernel = nullptr;
    std::shared_ptr<cl::Program> _program = nullptr;

    int _blockSizeX = 1;
    int _blockSizeY = 1;
    int _samples = -1;
    int _height = -1;
    int _width = -1;

    /**
     * Compile kernels, initialize buffers, calculate work-group sizes.
     * Sets _blockSizeX, blockSizeY.
     */
    int InitContext();

    /**
     * Create all buffers. Initialize _randBuffer with random values and copy it to the device.
     * Also sets _samples, _height and _width;
     */
    void InitData();

    /**
     * Set the kernel arguments using the buffers and constants calculate so far.
     */
    void SetKernelArguments();

    /**
     * Enqueue kernel and wait for result.
     * Executed multiple times.
     * Also prints statistics
     */
    void ExecuteKernels();

    /**
     * Cleanup all buffers, kernels and the program instance.
     */
    void Cleanup();

    /**
     * Set the correct work group size.
     * Call all functions declared above in correct order.
     */
    void RunInternal();

public:
    explicit BlackScholes(std::shared_ptr<ComputeController> controller);
    virtual ~BlackScholes();

    /**
     * Execute the benchmark.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_BLACKSCHOLES_HPP
