#ifndef __BENCH_BENCHMARKS_STREAMCLUSTER_HPP
#define __BENCH_BENCHMARKS_STREAMCLUSTER_HPP

#include "../benchmarkbase.hpp"

#include <memory>

namespace benchmarks {

/**
 * Another clustering algorithm used in machine learning.
 */
class StreamCluster : public BenchmarkBase {
private:
    std::shared_ptr<cl::Buffer> _pointBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _coordinatesBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _costBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _centerTableBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _membershipBuffer = nullptr;
    std::shared_ptr<cl::Kernel> _kernel = nullptr;
    std::shared_ptr<cl::Program> _program = nullptr;

    /**
     * Compile kernel, create buffers and set kernel arguments.
     */
    template <typename TItem>
    int InitContext();

    /**
     * Initialize buffers and copy them to the target device.
     */
    template <typename TItem>
    void InitDeviceMemory();

    /**
     * Execute the kernel a couple of times and generate statistics.
     */
    void Execute();

    /**
     * Release buffers, kernels and program instance.
     */
    void Cleanup();

public:
    explicit StreamCluster(std::shared_ptr<ComputeController> controller);

    virtual ~StreamCluster();

    /**
     * Execute benchmark.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_STREAMCLUSTER_HPP
