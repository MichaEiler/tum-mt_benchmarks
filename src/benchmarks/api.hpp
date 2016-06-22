#ifndef __BENCH_BENCHMARKS_API_HPP
#define __BENCH_BENCHMARKS_API_HPP

#include "../benchmarkbase.hpp"

namespace benchmarks {

/**
 * The idea was to test the api overhead.
 * So far this benchmark only compares the time required
 * to build from source against compilation from a binary.
 */
class Api : public BenchmarkBase {
private:
    void CompilationTimeAnalysis(const std::string& kernelSourceFile);

public:
    explicit Api(std::shared_ptr<ComputeController> controller);

    virtual ~Api();

    /**
     * Execute tests.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_API_HPP
