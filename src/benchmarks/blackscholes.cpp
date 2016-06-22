#include "blackscholes.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int RANDOM_SEED = 85733;
static const int SAMPLES = 256 * 256 * 1024;//64;
static const int TEST_ITERATIONS = 100;

BlackScholes::BlackScholes(std::shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

BlackScholes::~BlackScholes() {

}

int BlackScholes::InitContext() {
    string compilerParams = GetCompilerFlags<float>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "blackscholes.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = CL_SUCCESS;
    _scalarKernel = make_shared<cl::Kernel>(*_program, "blackScholes_scalar", &status);
    CHECK_RETURN_ERROR(status);
    _vectorizedKernel = make_shared<cl::Kernel>(*_program, "blackScholes", &status);
    CHECK_RETURN_ERROR(status);

    // decide upon 2d-work-group size
    size_t scalarMaxWorkGroupSize = 0;
    status = _scalarKernel->getWorkGroupInfo(_controller->SelectedDevice(), CL_KERNEL_WORK_GROUP_SIZE, &scalarMaxWorkGroupSize);
    CHECK_RETURN_ERROR(status);
    size_t vectorizedMaxWorkGroupSize = 0;
    status = _vectorizedKernel->getWorkGroupInfo(_controller->SelectedDevice(), CL_KERNEL_WORK_GROUP_SIZE, &vectorizedMaxWorkGroupSize);
    CHECK_RETURN_ERROR(status);

    int maxWorkGroupSize = static_cast<int>(min(scalarMaxWorkGroupSize, vectorizedMaxWorkGroupSize));
    maxWorkGroupSize = min(_requestedWorkGroupSize, maxWorkGroupSize);

    _blockSizeX = 1;
    _blockSizeY = 1;

    while ( (2 * _blockSizeX * _blockSizeY) <= maxWorkGroupSize) {
        _blockSizeX <<= 1;
        if ((2 * _blockSizeX * _blockSizeY) <= maxWorkGroupSize)
            _blockSizeY <<= 1;
    }
    cout << "Work-Group-Size: " << _blockSizeX << "*" << _blockSizeY << endl;

    return 0;
}

void BlackScholes::InitData() {
    _samples = SAMPLES / 4;
    _samples = RoundToMultipleOf(_samples, _requestedWorkGroupSize);

    int side = static_cast<int>(sqrt(_samples));
    side = RoundToMultipleOf(side, _requestedWorkGroupSize);
    _samples = side * side;
    _height = _width = side;

    _randBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_ONLY, _samples * sizeof(cl_float4));
    _callPriceBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_WRITE_ONLY, _samples * sizeof(cl_float4));
    _putPriceBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_WRITE_ONLY, _samples * sizeof(cl_float4));

    default_random_engine randomEngine(RANDOM_SEED);
    uniform_real_distribution<float> valueDistribution(0.0, 1.0);

    vector<float> randData;
    randData.resize(_samples * 4);
    for (size_t i = 0; i < randData.size(); ++i) {
        randData[i] = valueDistribution(randomEngine);
    }

    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_randBuffer, CL_TRUE, 0, _samples * sizeof(cl_float4), &randData[0]);
    queue.finish();
}

void BlackScholes::SetKernelArguments() {
    cl_int scalarRowStride = 4 * _width;
    _scalarKernel->setArg(0, *_randBuffer);
    _scalarKernel->setArg(1, scalarRowStride);
    _scalarKernel->setArg(2, *_callPriceBuffer);
    _scalarKernel->setArg(3, *_putPriceBuffer);

    _vectorizedKernel->setArg(0, *_randBuffer);
    _vectorizedKernel->setArg(1, _width);
    _vectorizedKernel->setArg(2, *_callPriceBuffer);
    _vectorizedKernel->setArg(3, *_putPriceBuffer);
}

void BlackScholes::ExecuteKernels() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::NDRange localWorkSize(_blockSizeX, _blockSizeY);
    cl::NDRange globalWorkSizeScalar(4 * _width, _height);
    cl::NDRange globalWorkSizeVectorized(_width, _height);
    cl_int status = CL_SUCCESS;

    string testName = "BlackScholes (scalar),     ";
    PerformTest([&](cl::Event& event) -> void {
            status = queue.enqueueNDRangeKernel(*_scalarKernel, cl::NullRange, globalWorkSizeScalar, localWorkSize, nullptr, &event);
            CHECK(status);
        }, testName, TEST_ITERATIONS);

    testName = "BlackScholes (vectorized), ";
    PerformTest([&](cl::Event& event) -> void {
            status = queue.enqueueNDRangeKernel(*_vectorizedKernel, cl::NullRange, globalWorkSizeVectorized, localWorkSize, nullptr, &event);
            CHECK(status);
        }, testName, TEST_ITERATIONS);
}

void BlackScholes::Cleanup() {
    _randBuffer.reset();
    _callPriceBuffer.reset();
    _putPriceBuffer.reset();
    _scalarKernel.reset();
    _vectorizedKernel.reset();
    _program.reset();
}

void BlackScholes::RunInternal() {
    RequestWorkGroupSize(256); // 256 is the default in amd samples code

    if (InitContext() == 0) {
        InitData();
        SetKernelArguments();
        ExecuteKernels();
        Cleanup();
    }
}

void BlackScholes::Run() {
    RunInternal();
    cout << endl;
}
