#ifndef __BENCH_BENCHMARKS_MEMORY_HPP
#define __BENCH_BENCHMARKS_MEMORY_HPP

#include "../benchmarkbase.hpp"

#include <string>

namespace cl {
    class Buffer;
    class Event;
}

namespace benchmarks {

/**
 * Some micro-benchmarks to test memory performance.
 */
class Memory : public BenchmarkBase {
private:
    template<typename TItem>
    void FillBufferWithContent(TItem *buffer, int length);

    /**
     * Copy a memory buffer to the device.
     * 
     * @param align true if the memory on the host should be aligned efficiently.
     */
	void CopyMemoryToDevice(bool align);

    /**
     * Copy a regular buffer allocated with new or malloc (=unpinned) to the device.
     */
    void CopyUnpinnedMemoryToDevice();

    /**
     * Copy back to host memory.
     */
    void CopyToHostMemory();
    
    /**
     * Copy to unpinned host memory.
     */
    void CopyToUnpinnedHostMemory();

    /**
     * Let a kernel write to host memory.
     *
     * @param align true if the buffer on the host memory should be aligned efficiently.
     */
	void WriteToHostMemory(bool align);

    /**
     * Read from host memory.
     */
    void ReadFromHostMemory();

    /**
     * Compare column- vs row-major data layouts.
     */
    void AnalyseMemoryAccessPatterns();

    /**
     * returns the amount of bits to which the gpu aligns all their memory objects
     * 
     * @return device dependent alignment factor
     */
    int AlignmentFactor();

    /**
     * moves a pointer to an address which is aligned to the parameter alignment
     * Note: make sure that the size of the buffer to which address points is
     *     yourSize + alignment + displacement
     */
    template <typename TItem>
    TItem* AlignAddress(TItem* address, const int alignment, const int displacement);

public:
    explicit Memory(std::shared_ptr<ComputeController> controller)
        : BenchmarkBase(controller) {

    }

    virtual ~Memory() { }

    /**
     * Excute all tests described by the functions declared above.
     */
    void Run();
};

}

#endif // __BENCH_BENCHMARKS_MEMORY_HPP
