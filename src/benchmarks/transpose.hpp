#ifndef __BENCH_BENCHMARKS_TRANSPOSE_HPP
#define __BENCH_BENCHMARKS_TRANSPOSE_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * Transposes a matrix on an OpenCL device.
 */
class Transpose : public BenchmarkBase {
private:
    std::shared_ptr<cl::Buffer> _inputBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _outputBuffer1 = nullptr;
    std::shared_ptr<cl::Buffer> _outputBuffer2 = nullptr;
    std::shared_ptr<cl::Kernel> _transposeSimpleKernel = nullptr;
    std::shared_ptr<cl::Kernel> _transposeOptimizedKernel = nullptr;
    std::shared_ptr<cl::Program> _program = nullptr;

    template <typename TItem>
    void GenerateInputData(int width, int height, TItem *buffer);

    /**
     * Compile kernel, create buffers, initialize data.
     */
    int InitContext();

    /**
     * Compare the result of both Run*() methods.
     */
    void Validate();

    /**
     * Execute a naive implementation of transpose.
     */
    void RunSimple();

    /**
     * Execute an optimized version of tranpose. 
     * (It uses local memory and work-group barriers).
     */
    void RunOptimized();

public:
    explicit Transpose(std::shared_ptr<ComputeController> controller);

    virtual ~Transpose();

    /**
     * Execute benchmark.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_TRANSPOSE_HPP
