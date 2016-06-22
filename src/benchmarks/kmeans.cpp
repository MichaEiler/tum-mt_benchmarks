#include "kmeans.hpp"

#include <cstring> // for memset only
#include <fstream>
#include <iostream>
#include <sstream>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int NUMBER_OF_CLUSTERS = 5;
static const int ALGORITHM_ITERATIONS = 50;               // number of iterations per cluster (max iterations in rodinia is 500)
static const string KMEANS_DATASET = "kmeans/kdd_cup";

KMeans::KMeans(std::shared_ptr<ComputeController> controller) 
    : BenchmarkBase(controller) {

}

KMeans::~KMeans() {

}

/*
 * This functions parses the inptu file use to calculate the clusters.
 */
template <typename TItem>
TItem* KMeans::LoadInputData() {
    ifstream fileStream(CL_DATA_PATH_PREFIX + KMEANS_DATASET, ios::in);
    string inputData = "";

    fileStream.seekg(0, ios::end);
    inputData.resize(static_cast<size_t>(fileStream.tellg()));
    fileStream.seekg(0, ios::beg);
    fileStream.read(&inputData[0], inputData.size());
    fileStream.close();

    stringstream stream(inputData, ios::in);

    stream >> _pointCount;
    stream >> _featureCount;

    if (!stream.good())
        return nullptr;

    // allocate memory for feature data
    TItem *features = new TItem[_pointCount * _featureCount];

    bool failed = false;
    for (int p = 0; p < _pointCount && !failed; ++p) {
        for (int f = 0; f < _featureCount; ++f){
            float input = 0.0;
            stream >> input;

            if (!stream.good()) {
                failed = true;
                break;
            }

            features[p * _featureCount + f] = static_cast<TItem>(input);
        }
    }

    if (failed) {
        _pointCount = -1;
        _featureCount = -1;
        delete[] features;
        return nullptr;
    }
    _features = static_cast<void*>(features);
    return features;
}

/*
 * This function intializes all objects related to OpenCL (e.g. program, kernels, buffers).
 * It also uses a kernel function to transpose the input data so that we don't have to do this later in the benchmark code.
 */
template <typename TItem>
int KMeans::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "kmeans.cl", compilerParams);

    if (_program.get() == nullptr) {
        CleanupContext<TItem>();
        return -1;
    }

    cl_int status = 0;
    _transposeKernel = make_shared<cl::Kernel>(*_program, "kmeans_transpose", &status);
    CHECK_RETURN_ERROR(status);

    _kmeansColumnMajorKernel = make_shared<cl::Kernel>(*_program, "kmeans_kernel_col", &status);
    CHECK_RETURN_ERROR(status);

    _kmeansRowMajorKernel = make_shared<cl::Kernel>(*_program, "kmeans_kernel_row", &status);
    CHECK_RETURN_ERROR(status);

    // allocate buffers
    const size_t bufferSize = _pointCount * _featureCount * sizeof(TItem);
    _valuesRowBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _valuesColBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _clusterBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_CLUSTERS * _featureCount * sizeof(TItem));
    _membershipBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _pointCount * sizeof(cl_int));

    // write input data into inputDataRowMajor
    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_valuesRowBuffer, CL_TRUE, 0, bufferSize, _features);

    _workGroupSize = _requestedWorkGroupSize;
    _workItemCount = _pointCount;
    _workItemCount = RoundToMultipleOf(_pointCount, _requestedWorkGroupSize);

    return 0;
}

void KMeans::SetKernelParameters() {
    // tranpose data for column-major kmeans
    cl_int pointCountCl = static_cast<cl_int>(_pointCount);
    cl_int featureCountCl = static_cast<cl_int>(_featureCount);
    _transposeKernel->setArg(0, *_valuesRowBuffer);
    _transposeKernel->setArg(1, *_valuesColBuffer);
    _transposeKernel->setArg(2, pointCountCl);
    _transposeKernel->setArg(3, featureCountCl);

    // set arguments for other kernels
    cl_int clustersCl = NUMBER_OF_CLUSTERS;
    _kmeansColumnMajorKernel->setArg(0, *_valuesColBuffer);
    _kmeansColumnMajorKernel->setArg(1, *_clusterBuffer);
    _kmeansColumnMajorKernel->setArg(2, *_membershipBuffer);
    _kmeansColumnMajorKernel->setArg(3, pointCountCl);
    _kmeansColumnMajorKernel->setArg(4, clustersCl);
    _kmeansColumnMajorKernel->setArg(5, featureCountCl);

    _kmeansRowMajorKernel->setArg(0, *_valuesRowBuffer);
    _kmeansRowMajorKernel->setArg(1, *_clusterBuffer);
    _kmeansRowMajorKernel->setArg(2, *_membershipBuffer);
    _kmeansRowMajorKernel->setArg(3, pointCountCl);
    _kmeansRowMajorKernel->setArg(4, clustersCl);
    _kmeansRowMajorKernel->setArg(5, featureCountCl);
}

/*
 * Transpose inputData from row-major to column-major variant.
 */
void KMeans::TransposeInputData() {
    cl::CommandQueue& queue = _controller->Queue();

    // run gpu code for transpose, reading data back to host not necessary
    int64_t totalTimeCPU = 0;
    int64_t totalTimeGPU = 0;

    cl::NDRange global(_workItemCount);
    cl::NDRange local(_workGroupSize);
    cl::Event event;
    cl_ulong startTime, endTime;
    cl_int status;

    for (int i = 0; i < 50; ++i) { // do this 50 times to create realistic time values        
        queue.flush();
        _timer.Remember();
        status = queue.enqueueNDRangeKernel(*_transposeKernel, cl::NullRange, global, local, nullptr, &event);
        WAIT_AND_CHECK(event, status);
        queue.finish();

        totalTimeCPU += _timer.Diff();
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        totalTimeGPU += (endTime - startTime);
    }

    cout << "Transpose, CPU: " << totalTimeCPU / 50 << ", GPU: " << totalTimeGPU / 50 << endl;
}

/*
 * Cleanup all resources allocated by InitContext().
 */
template <typename TItem>
void KMeans::CleanupContext() {
    _transposeKernel.reset();
    _kmeansColumnMajorKernel.reset();
    _kmeansRowMajorKernel.reset();
    _valuesColBuffer.reset();
    _valuesRowBuffer.reset();
    _clusterBuffer.reset();
    _membershipBuffer.reset();
    _program.reset();

    if (_features != nullptr) {
        TItem *features = static_cast<TItem*>(_features);
        _features = nullptr;
        delete[] features;
    }

    _featureCount = -1;
    _pointCount = -1;
}

template <typename TItem, typename TInteger>
void KMeans::UpdateClusterPositions(vector<TItem>& clusters, vector<TItem>& centerValues, vector<int>& pointsPerCluster, vector<TInteger>& membership) {
    TItem *features = static_cast<TItem*>(_features);

    for (int p = 0; p < _pointCount; ++p) {
        int clusterId = membership[p];
        pointsPerCluster[clusterId]++;
        for (int f = 0; f < _featureCount; ++f) {
            centerValues[clusterId * _featureCount + f] += features[p * _featureCount + f];
        }
    }

    for (int c = 0; c < NUMBER_OF_CLUSTERS; ++c) {
        for (int f = 0; f < _featureCount; ++f) {
            if (pointsPerCluster[c] > 0)
                clusters[c * _featureCount + f] = centerValues[c * _featureCount + f] / pointsPerCluster[c];
            centerValues[c * _featureCount + f] = 0.0;
        }
        pointsPerCluster[c] = 0;
    }
}

template <typename TItem>
void KMeans::RunInternal(bool columnMajor) {

    if (LoadInputData<TItem>() == nullptr) {
        cerr << "Failed to load input data!" << endl;
        return;
    }

    if (InitContext<TItem>() != 0) {
        return;
    }

    SetKernelParameters();
    if (columnMajor)
        TransposeInputData();

    shared_ptr<cl::Kernel> kmeansKernel = columnMajor ? _kmeansColumnMajorKernel : _kmeansRowMajorKernel;
    if (kmeansKernel.get() == nullptr || _features == nullptr) {
        CleanupContext<TItem>();
        return;
    }

    TItem *features = static_cast<TItem*>(_features);
    vector<int> pointsPerCluster(NUMBER_OF_CLUSTERS, 0);
    vector<TItem> centerValues(NUMBER_OF_CLUSTERS * _featureCount, 0);
    vector<TItem> clusters;

    clusters.resize(NUMBER_OF_CLUSTERS * _featureCount);

    // initialization of clusters sets every cluster for every single feature to a not really random point
    for (int c = 0; c < NUMBER_OF_CLUSTERS; ++c) {
        for (int f = 0; f < _featureCount; ++f) {
            int pointIndex = (c * 85733 + f * 83) % _pointCount; // try to generate more or less random point index
            clusters[c * _featureCount + f] = features[pointIndex * _featureCount + f];
        }
    }

    // membership container, store the information to which cluster a point belongs
    vector<int> membershipHost(_pointCount, 0);

    cl::CommandQueue& queue = _controller->Queue();

    int64_t totalTimeCPU = 0;
    int64_t totalTimeGPU = 0;

    cl::NDRange global(_workItemCount);
    cl::NDRange local(_workGroupSize);
    cl::Event event;
    cl_ulong startTime, endTime;
    cl_int status;

    for (int i = 0; i < ALGORITHM_ITERATIONS; ++i) { // in rodinia they use an additional threshold -> but wrong implementation results in infinity loop

        // copy new cluster information to device
        queue.enqueueWriteBuffer(*_clusterBuffer, CL_TRUE, 0, NUMBER_OF_CLUSTERS * _featureCount * sizeof(TItem), &clusters[0]);

        // execute kernel code
        queue.flush();
        _timer.Remember();
        status = queue.enqueueNDRangeKernel(*kmeansKernel, cl::NullRange, global, local, nullptr, &event);
        if (status != CL_SUCCESS) {
            cerr << "Error " << status << " in " << __FILE__ << " on line: " << __LINE__ << endl;
            break;
        }
        event.wait();
        queue.flush();

        totalTimeCPU += _timer.Diff();
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        totalTimeGPU += (endTime - startTime);

        // read back result
        queue.enqueueReadBuffer(*_membershipBuffer, CL_TRUE, 0, _pointCount * sizeof(cl_int), &membershipHost[0]);
        queue.flush();

        UpdateClusterPositions<TItem, cl_int>(clusters, centerValues, pointsPerCluster, membershipHost);
    }

    cout << (columnMajor ? "Col-Major" : "Row-Major");
    cout << ", CPU: " << totalTimeCPU << ", GPU: " << totalTimeGPU << endl;

    // cleanup
    CleanupContext<TItem>();
}

void KMeans::Run() {
    int workGroupSizes[] = { 32, 64, 128, 192, 256, 512 };

    for (auto& workGroupSize : workGroupSizes) {
        RequestWorkGroupSize(workGroupSize);

        cout << "Running: kmeans<float>,  WorkGroupSize: " << workGroupSize << endl;
        RunInternal<float>(true);
        RunInternal<float>(false);

        if (_controller->SupportsDoublePrecision()) {
            cout << "Running: kmeans<double>, WorkGroupSize: " << workGroupSize << endl;
            RunInternal<double>(true);
            RunInternal<double>(false);
        }

    }

    cout << endl;
}
