#ifndef __BENCH_BENCHMARKS_STENCIL_HPP
#define __BENCH_BENCHMARKS_STENCIL_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * Applies a stencil on an matrix.
 */
class Stencil : public BenchmarkBase {
private:

	std::shared_ptr<cl::Buffer> _sourceBuffer = nullptr;
	std::shared_ptr<cl::Buffer> _inputBuffer = nullptr;
	std::shared_ptr<cl::Buffer> _outputBuffer = nullptr;
	std::shared_ptr<cl::Kernel> _stencilKernel = nullptr;
	std::shared_ptr<cl::Program> _program = nullptr;

    /**
     * Execute the kernels and generate statistics.
     */
    template <typename TItem>
    void RunInternal();

    /**
     * Fill the given matrix with some numbers and copy it to the device.
     */
    template <typename TItem>
    void FillMatrix(cl::Buffer &buffer, int width, int height);

    /**
     * Compile kernel.Initialize buffer containing matrix.
     */
    template <typename TItem>
    int InitContext();

public:
    explicit Stencil(std::shared_ptr<ComputeController> controller);

    virtual ~Stencil();

    /**
     * Execute benchmarks with SPFP and DPFP.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_STENCIL_HPP
