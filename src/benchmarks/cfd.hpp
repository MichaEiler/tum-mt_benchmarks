#ifndef __BENCH_BENCHMARKS_CFD_HPP
#define __BENCH_BENCHMARKS_CFD_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * The benchmarks works with fluid dynamics.
 */
class Cfd : public BenchmarkBase {
private:
    std::shared_ptr<cl::Program> _program = nullptr;
    std::shared_ptr<cl::Kernel> _memsetKernel = nullptr;
    std::shared_ptr<cl::Kernel> _initializeVariablesKernel = nullptr;
    std::shared_ptr<cl::Kernel> _computeStepFactorKernel = nullptr;
    std::shared_ptr<cl::Kernel> _computeFluxKernel = nullptr;
    std::shared_ptr<cl::Kernel> _timeStepKernel = nullptr;

    std::shared_ptr<cl::Buffer> _ff_variableBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _ff_fluxXBuffer = nullptr; // far field, flux contribution momentum, X axis
    std::shared_ptr<cl::Buffer> _ff_fluxYBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _ff_fluxZBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _ff_fluxEnergyBuffer = nullptr; // far field flux contribution density energy
    std::shared_ptr<cl::Buffer> _areasBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _surroundingElementsCountersBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _normalVectorsBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _variablesBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _oldVariablesBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _fluxesBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _stepFactorsBuffer = nullptr;

    int _pointCount = 0;
    int _pointCountPadded = 0;

    float *_areas = nullptr;
    int *_surroundingElementsCounters = nullptr;
    float *_normalVectors = nullptr;

    void InitFarFieldData();

    /**
     * Load some input data from the data/cfd/ directory.
     */
    bool LoadInputData();

    /**
     * Initializes all buffers and compiles the kernel.
     */
    int InitKernelsAndBuffers();

    /**
     * Sets the kernel arguments.
     */
    void SetKernelArguments();

    /**
     * Initialize the memory on the device by using additional kernels.
     */
    void InitDeviceMemory();

    /**
     * Execute _computeStepFactorKernel, _compueFluxKernel and _timeStepKernel
     * to simulate fluid dyanmics. Also does some statistical output.
     */
    void RunInternal();

    /**
     * Cleanup buffers, kernels and program instance.
     */
    void Cleanup();

public:
    explicit Cfd(std::shared_ptr<ComputeController> controller);

    virtual ~Cfd();

    /**
     * Execute benchmark.
     */
    void Run();

    Cfd(const Cfd&) = delete;
    Cfd& operator=(const Cfd&) = delete;
    Cfd(const Cfd&&) = delete;
    Cfd& operator=(const Cfd&&) = delete;
};

}

#endif // __BENCH_BENCHMARKS_CFD_HPP
