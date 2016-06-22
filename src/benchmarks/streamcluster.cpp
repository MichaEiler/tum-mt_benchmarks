#include "streamcluster.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int MAXIMUM_CLUSTERS = 5;
static const int NUMBER_OF_POINTS = 1024 * 1024;
static const int POINT_DIMENSION = 3;
static const int TEST_ITERATIONS = 100;
static const int RANDOM_SEED = 85733;

template <typename TItem>
struct Point {
    TItem weight;
    int assign;
    TItem cost;
};

StreamCluster::StreamCluster(std::shared_ptr<ComputeController> controller)
        : BenchmarkBase(controller) {

}

StreamCluster::~StreamCluster() {

}

template <typename TItem>
int StreamCluster::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "streamcluster.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = CL_SUCCESS;
    _kernel = make_shared<cl::Kernel>(*_program, "pgain_kernel", &status);
    CHECK_RETURN_ERROR(status);

    _pointBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_POINTS * sizeof(Point<TItem>));
    _coordinatesBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_POINTS * POINT_DIMENSION * sizeof(TItem));
    _costBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_POINTS * (MAXIMUM_CLUSTERS + 1) * sizeof(TItem));
    _centerTableBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_POINTS * sizeof(int));
    _membershipBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NUMBER_OF_POINTS * sizeof(char));

    cl_int numberOfPoints = NUMBER_OF_POINTS;
    cl_int numberOfCenters = MAXIMUM_CLUSTERS;

    _kernel->setArg(0, *_pointBuffer);
    _kernel->setArg(1, *_coordinatesBuffer);
    _kernel->setArg(2, *_costBuffer);
    _kernel->setArg(3, *_centerTableBuffer);
    _kernel->setArg(4, *_membershipBuffer);
    _kernel->setArg(5, numberOfPoints);
    _kernel->setArg(7, numberOfCenters);

    return 0;
}

void StreamCluster::Cleanup() {
    _pointBuffer.reset();
    _coordinatesBuffer.reset();
    _costBuffer.reset();
    _centerTableBuffer.reset();
    _membershipBuffer.reset();

    _kernel.reset();
    _program.reset();
}

template <typename TItem>
void StreamCluster::InitDeviceMemory() {
    // initialize datastructures
    vector<TItem> coordinates;
    coordinates.resize(NUMBER_OF_POINTS * POINT_DIMENSION);
    for (int i = 0; i < NUMBER_OF_POINTS * POINT_DIMENSION; ++i) {
        coordinates[i] = static_cast<TItem>(i) / INT32_MAX;
    }

    default_random_engine randomEngine(RANDOM_SEED);
    uniform_real_distribution<TItem> weightDistribution(static_cast<TItem>(0.7), static_cast<TItem>(1.3));
    uniform_int_distribution<int> pointDistribution(0, NUMBER_OF_POINTS - 1);

    vector<Point<TItem>> points;
    points.resize(NUMBER_OF_POINTS);
    for (int i = 0; i < NUMBER_OF_POINTS; ++i) {
        points[i].weight = weightDistribution(randomEngine); // should be value between 0.7 and 1.3
        points[i].assign = pointDistribution(randomEngine); // another point id (random?)
        points[i].cost = weightDistribution(randomEngine);
    }

    vector<int> centerTable;
    centerTable.resize(NUMBER_OF_POINTS, 0);
    int count = 0;
    for (int i = 0; i < NUMBER_OF_POINTS && count < MAXIMUM_CLUSTERS; ++i) {
        if ( (i % 2) == 0 ) {
            centerTable[i] = count++; 
        }
    }

    vector<char> zeroBuffer(NUMBER_OF_POINTS * MAXIMUM_CLUSTERS * sizeof(TItem), 0);

    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_membershipBuffer, CL_TRUE, 0, NUMBER_OF_POINTS * sizeof(char), &zeroBuffer[0]);
    queue.enqueueWriteBuffer(*_costBuffer, CL_TRUE, 0, NUMBER_OF_POINTS * MAXIMUM_CLUSTERS * sizeof(TItem), &zeroBuffer[0]);
    queue.enqueueWriteBuffer(*_coordinatesBuffer, CL_TRUE, 0, NUMBER_OF_POINTS * POINT_DIMENSION * sizeof(TItem), &coordinates[0]);
    queue.enqueueWriteBuffer(*_pointBuffer, CL_TRUE, 0, NUMBER_OF_POINTS * sizeof(Point<TItem>), &points[0]);
    queue.enqueueWriteBuffer(*_centerTableBuffer, CL_TRUE, 0, NUMBER_OF_POINTS * sizeof(int), &centerTable[0]);
    queue.finish();
}

void StreamCluster::Execute() {
    default_random_engine randomEngine(RANDOM_SEED);
    uniform_int_distribution<int> pointDistribution(0, NUMBER_OF_POINTS - 1);

    cl::CommandQueue& queue = _controller->Queue();
    cl::NDRange local(_requestedWorkGroupSize);
    cl::NDRange global(RoundToMultipleOf(NUMBER_OF_POINTS, _requestedWorkGroupSize));
    cl::Event event;
    cl_ulong startTime, endTime;

    _cpuStatistics.Clear();
    _gpuStatistics.Clear();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        cl_long randomPointId = pointDistribution(randomEngine);
        _kernel->setArg(6, randomPointId);    // use a random point, in the original algorithm 
                                              // this iterates over a list of feasible points

        _timer.Remember();
        cl_int status = queue.enqueueNDRangeKernel(*_kernel, cl::NullRange, global, local, nullptr, &event);
        WAIT_AND_CHECK(event, status);
        _cpuStatistics.Add(_timer.Diff());

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        _gpuStatistics.Add(endTime - startTime);
    }

    // print sum instead of mean/deviation since that was what we started with
    cout << " CPU: " << _cpuStatistics.Sum() << ", GPU: " << _gpuStatistics.Sum() << endl;
}

void StreamCluster::Run() {
    cout << "StreamCluster-Test:" << endl;

    int workGroupSizes[] = { 32, 64, 128, /*192,*/ 256, 512 };

    for (auto& workGroupSize : workGroupSizes) {
        RequestWorkGroupSize(workGroupSize);

        cout << "StreamCluster<float>,  WorkGroupSize: " << workGroupSize << ", ";
        if (InitContext<float>() == 0) {
            InitDeviceMemory<float>();
            Execute();
            Cleanup();
        }

        if (_controller->SupportsDoublePrecision()) {
            cout << "StreamCluster<double>, WorkGroupSize: " << workGroupSize << ", ";
            if (InitContext<double>() == 0) {
                InitDeviceMemory<double>();
                Execute();
                Cleanup();
            }
        }
    }

    cout << endl;
}
