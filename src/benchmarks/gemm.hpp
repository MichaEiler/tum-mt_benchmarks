#ifndef __BENCH_BENCHMARKS_GEMM_HPP
#define __BENCH_BENCHMARKS_GEMM_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * This class implements a matrix multiplication benchmark.
 * The kernel code used in it is stored in src/cl/gemm.cl.
 */
class Gemm : public BenchmarkBase {
private:
    std::shared_ptr<cl::Buffer> _deviceMatrixA = nullptr;
    std::shared_ptr<cl::Buffer> _deviceMatrixB = nullptr;
    std::shared_ptr<cl::Buffer> _deviceMatrixC = nullptr;
    std::shared_ptr<cl::Buffer> _sourceMatrixA = nullptr;
    std::shared_ptr<cl::Buffer> _sourceMatrixB = nullptr;
    std::shared_ptr<cl::Buffer> _sourceMatrixC = nullptr;
    std::shared_ptr<cl::Kernel> _nnKernel = nullptr;
    std::shared_ptr<cl::Kernel> _ntKernel = nullptr;
    std::shared_ptr<cl::Program> _program = nullptr;

    int _bufferSize = -1;

    /**
     * Initializes all buffers and compiles the kernel code.
     * The type parameter is used to decide whether the single-
     * or double precision FP or INT version should be executed.
     *
     * @return zero on success
     */
    template <typename TItem>
    int InitContext();

    /**
     * Sets the correct buffer and size values as arguments of the kernel instances.
     */
    template <typename TItem>
    void SetKernelArguments();

    /**
     * Filles the buffers (matrices) with pseudo-random data.
     */
    template <typename TItem>
    void InitData();

    /**
     * Enqueues the kernel and therefore executes the actual
     * matrix multiplication on the OpenCL device.
     */
    void ExecuteKernels();

    /**
     * Frees all buffers, kernels and the program instance.
     */
    void Cleanup();

    /**
     * A wrapper around the functions declared above.
     * Simply calls them in appropriate order.
     */
    template <typename TItem>
    void RunInternal();

public:
    explicit Gemm(std::shared_ptr<ComputeController> controller)
        : BenchmarkBase(controller) {

    }

    virtual ~Gemm() { }

    /**
     * Execute all version of the matrix multiplication benchmark.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_GEMM_HPP
