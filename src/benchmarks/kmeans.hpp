#ifndef __BENCH_BENCHMARKS_KMEANS_HPP
#define __BENCH_BENCHMARKS_KMEANS_HPP

#include "../benchmarkbase.hpp"

#include <vector>

namespace benchmarks {

/**
 * Clustering algorithm used in e.g. machine learning.
 */
class KMeans : public BenchmarkBase {
private:
    std::shared_ptr<cl::Program> _program = nullptr;
    std::shared_ptr<cl::Kernel> _transposeKernel = nullptr;
    std::shared_ptr<cl::Kernel> _kmeansColumnMajorKernel = nullptr;
    std::shared_ptr<cl::Kernel> _kmeansRowMajorKernel = nullptr;
    std::shared_ptr<cl::Buffer> _valuesColBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _valuesRowBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _clusterBuffer = nullptr;
    std::shared_ptr<cl::Buffer> _membershipBuffer = nullptr;

    void *_features = nullptr;
    int _featureCount = -1;
    int _pointCount = -1;
    int _workGroupSize = 256;
    int _workItemCount = -1;

    /**
     * load the test data
     */
    template <typename TItem>
    TItem* LoadInputData();

    /**
     * initialize all objects related to OpenCL, program, kernels, buffers,...
     */
    template <typename TItem>
    int InitContext();

    /**
     * set the parameters for all OpenCL kernels
     */
    void SetKernelParameters();

    /**
     * convert test data from row-major to column-major
     */
    void TransposeInputData();

    /**
     * used to update the positions of all clusters after every single iteration of kmeans
     */
    template <typename TItem, typename TInteger>
    void UpdateClusterPositions(std::vector<TItem>& clusters, std::vector<TItem>& centerValues, 
        std::vector<int>& pointsPerCluster, std::vector<TInteger>& membership);

    /**
     * run the actual tests
     */
    template <typename TItem>
    void RunInternal(bool columnMajor);

    /**
     * cleanup OpenCL context and memory
     */
    template <typename TItem>
    void CleanupContext();

public:
    explicit KMeans(std::shared_ptr<ComputeController> controller);

    virtual ~KMeans();

    /**
     * Execute benchmark.
     */
    void Run();

    KMeans(const KMeans&) = delete;
    KMeans& operator=(const KMeans&) = delete;
    KMeans(const KMeans&&) = delete;
    KMeans& operator=(const KMeans&&) = delete;
};

}

#endif // __BENCH_BENCHMARKS_KMEANS_HPP
