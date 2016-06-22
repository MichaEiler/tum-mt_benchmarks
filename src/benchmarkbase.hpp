#ifndef __BENCH_BENCHMARKBASE_HPP
#define __BENCH_BENCHMARKBASE_HPP

#include "statistics.hpp"
#include "timer.hpp"

#include <functional>
#include <memory>
#include <string>
#include <typeinfo>

class ComputeController;

namespace cl {
    class Buffer;
    class Event;
    class Kernel;
    class Program;
}

namespace benchmarks {

/**
 * This class provides base functionallity shared by all benchmarks.
 */
class BenchmarkBase {
private:
    std::string GetCompilerFlagsInternal(const std::type_info& typeInfo);

protected:
    std::shared_ptr<ComputeController> _controller;
    Timer _timer;
    Statistics<int64_t> _cpuStatistics;
    Statistics<int64_t> _gpuStatistics;

    int _requestedWorkGroupSize;

    bool _optimizeForSpeed;
    bool _disableOptimization;

    template <typename TItem>
    std::string GetCompilerFlags() { return GetCompilerFlagsInternal(typeid(TItem)); }

    /**
     * A small helper function to test the performance of OpenCL kernels.
     * It simply calls the kernel code, waits for it to finish and analyses the performs.
     * It executes this a couple of times and then generates the statistics using the Statistics class.
     *
     * @param testFunction functiont o the actual OpenCL kernel enqueu command, it is important that
     *                     the cl::Event instance is used in the enqueue command, otherwise waiting for
     *                     it to finish will nto work
     * @param testName name of the test
     * @param iterations amount of times to execute the kernel before generating the statistics.
     * 
     * Example:
     *      PerformTest([&](cl::Event& event) -> void {
     *              queue.enqueueNDRangeKernel(someKernel, cl::NullRange, globalWorkItemCount, cl::NullRange, nullptr, &event);
     *          }, testName, ITERATIONS);
     */
    void PerformTest(std::function<void(cl::Event&)> testFunction, const std::string& testName, const int iterations);

    int RoundToPowerOf2(int i, int powerOf2);

    /**
     * Use this to calculate a valid number for the work-item count.
     * 
     * @param i number of work items
     * @param p base number
     * @return i + ((p - (i%p)) % p)
     */
    int RoundToMultipleOf(int i, int p);
public:
    explicit BenchmarkBase(std::shared_ptr<ComputeController> controller);
    virtual ~BenchmarkBase();

    /**
     * Adds -cl-opt-disable as flag to the compiler parameters.
     */
    void RequestDisableOptimization(bool disable) { _disableOptimization = disable; }

    /**
     * Adds -cl-fast-relaxed-math and -cl-no-signed-zeros to the compiler parameters.
     */
    void RequestOptimizationForSpeed(bool speed) { _optimizeForSpeed = speed; }

    /**
     * Sets the preferred work-group size for a benchmark.
     * Still depends on the actual benchmark whether it uses the suggestion.
     */
    void RequestWorkGroupSize(int workGroupSize);

    /**
     * Function which executes the actual benchmarks. Must be implemented by all sub-classes.
     */
    virtual void Run() = 0;
};

}

#endif // __BENCH_BENCHMARKBASE_HPP
