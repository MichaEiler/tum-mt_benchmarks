#ifndef __BENCH_BENCHMARKS_VECOP_HPP
#define __BENCH_BENCHMARKS_VECOP_HPP

#include "../benchmarkbase.hpp"

namespace benchmarks {

class Vecop : public BenchmarkBase {
private:
	/**
	 * Execute the kernel described by vectorOperation.
	 *
	 * @param vectorOperation the name of the kernel (=operation) to be executed on the device
	 */
    template <typename TItem>
    void RunInternal(const std::string& vectorOperation);

public:
    explicit Vecop(std::shared_ptr<ComputeController> controller)
        : BenchmarkBase(controller) {

    }

    virtual ~Vecop() { }

    /**
     * Execute vecadd, vecdiv and vecmul operations with double, float, int and long data types.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_VECOP_HPP
