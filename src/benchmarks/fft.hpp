#ifndef __BENCH_BENCHMARKS_FFT_HPP
#define __BENCH_BENCHMARKS_FFT_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * Fast Fourier Transform
 */
class Fft : public BenchmarkBase {
private:
	std::shared_ptr<cl::Buffer> _processBufferDevice = nullptr;
	std::shared_ptr<cl::Buffer> _validationBufferDevice = nullptr;
	std::shared_ptr<cl::Kernel> _checkKernel = nullptr;
	std::shared_ptr<cl::Kernel> _forwardKernel = nullptr;
	std::shared_ptr<cl::Kernel> _inverseKernel = nullptr;
	std::shared_ptr<cl::Program> _program = nullptr;

	int _totalBufferSize = -1;
	int _blockSize = -1;
	int _blocksToProcessHalf = -1;
	int _blocksToProcess = -1;
	int _numberOfFFTValuesHalf = -1;

	/**
	 * Initialize program instance and kernels.
	 */
	template <typename TItem>
	int InitContext();

	/**
	 * Generate input data.
	 */
	template <typename TItem>
	void InitData();

	/**
	 * Execute all kernels. FFT -> Inverse FFT and CHECK
	 */
	void ExecuteKernels();

	/**
	 * Release all resources.
	 */
	void Cleanup();

	/**
	 * Execute all functions above in order and with the same type.
	 */
    template <typename TItem>
    void RunInternal();

public:
    explicit Fft(std::shared_ptr<ComputeController> controller)
        : BenchmarkBase(controller) {

    }

    virtual ~Fft() { }

    /**
     * Execute benchmark once with a single-precision
     * FP and once with a double-precision FP parameter.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_FFT_HPP
