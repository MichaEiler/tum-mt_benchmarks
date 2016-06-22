#include "gemm.hpp"

#include <iostream>
#include <random>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const float  ALPHA = -1.0f;
static const float  BETA = 1.0f;
static const int    MATRIX_SIZE = 1024;
static const int    PASSES = 10;

template <typename TItem>
int Gemm::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "gemm.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = 0;
    _nnKernel = make_shared<cl::Kernel>(*_program, "sgemmNN", &status);
    CHECK_RETURN_ERROR(status);
    _ntKernel = make_shared<cl::Kernel>(*_program, "sgemmNT", &status);
    CHECK_RETURN_ERROR(status);

    _bufferSize = sizeof(TItem) * MATRIX_SIZE * MATRIX_SIZE;
    cl::Context& context = _controller->Context();
    _sourceMatrixA = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, _bufferSize);
    _sourceMatrixB = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, _bufferSize);
    _sourceMatrixC = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, _bufferSize);

    _deviceMatrixA = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, _bufferSize); 
    _deviceMatrixB = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, _bufferSize); 
    _deviceMatrixC = make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, _bufferSize);

    return 0;
}

template <typename TItem>
void Gemm::SetKernelArguments() {
    TItem alpha = static_cast<TItem>(ALPHA);
    TItem beta = static_cast<TItem>(BETA);

    _nnKernel->setArg(0, *_deviceMatrixA);
    _nnKernel->setArg(1, MATRIX_SIZE);
    _nnKernel->setArg(2, *_deviceMatrixB);
    _nnKernel->setArg(3, MATRIX_SIZE);
    _nnKernel->setArg(4, *_deviceMatrixC);
    _nnKernel->setArg(5, MATRIX_SIZE);
    _nnKernel->setArg(6, MATRIX_SIZE);
    _nnKernel->setArg(7, alpha);
    _nnKernel->setArg(8, beta);

    _ntKernel->setArg(0, *_deviceMatrixA);
    _ntKernel->setArg(1, MATRIX_SIZE);
    _ntKernel->setArg(2, *_deviceMatrixB);
    _ntKernel->setArg(3, MATRIX_SIZE);
    _ntKernel->setArg(4, *_deviceMatrixC);
    _ntKernel->setArg(5, MATRIX_SIZE);
    _ntKernel->setArg(6, MATRIX_SIZE);
    _ntKernel->setArg(7, alpha);
    _ntKernel->setArg(8, beta);
}

template <typename TItem>
void Gemm::InitData() {
    cl::CommandQueue& queue = _controller->Queue();
    TItem *A = static_cast<TItem*>(queue.enqueueMapBuffer(*_sourceMatrixA, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, _bufferSize));
    TItem *B = static_cast<TItem*>(queue.enqueueMapBuffer(*_sourceMatrixB, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, _bufferSize));
    TItem *C = static_cast<TItem*>(queue.enqueueMapBuffer(*_sourceMatrixC, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, _bufferSize));

    random_device randomDevice;
    default_random_engine engine(randomDevice());
    std::uniform_real_distribution<double> dist(0, 1);

    if (typeid(cl_int) != typeid(TItem) && typeid(cl_long) != typeid(TItem)) {
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
            A[i] = static_cast<TItem>(0.5 + dist(engine)*1.5);
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
            B[i] = static_cast<TItem>(0.5 + dist(engine)*1.5);
    } else {
        double maxValue = static_cast<double>(sizeof(TItem) == 4 ? INT32_MAX : INT64_MAX);

        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
            A[i] = static_cast<TItem>(dist(engine) * maxValue);
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
            B[i] = static_cast<TItem>(dist(engine) * maxValue);
    }        

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        C[i] = 0;

    queue.enqueueUnmapMemObject(*_sourceMatrixA, A);
    queue.enqueueUnmapMemObject(*_sourceMatrixB, B);
    queue.enqueueUnmapMemObject(*_sourceMatrixC, C);
}

void Gemm::ExecuteKernels() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::Event event;
    cl::NDRange localWorkSize(16, 4);
    cl::NDRange globalWorkSize( MATRIX_SIZE / 4, MATRIX_SIZE / 4 );
    cl_long endTime, startTime;
    cl_int status = CL_SUCCESS;

    uint64_t totalTimeCPU = 0;
    uint64_t totalTimeGPU = 0;

    status = queue.enqueueCopyBuffer(*_sourceMatrixC, *_deviceMatrixC, 0, 0, _bufferSize, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    for (auto& kernel : { _nnKernel, _ntKernel }) {
        for (int i = 0; i < PASSES; ++i) {
            status = queue.enqueueCopyBuffer(*_sourceMatrixA, *_deviceMatrixA, 0, 0, _bufferSize, nullptr, &event);
            WAIT_AND_CHECK(event, status);
            status = queue.enqueueCopyBuffer(*_sourceMatrixB, *_deviceMatrixB, 0, 0, _bufferSize, nullptr, &event);
            WAIT_AND_CHECK(event, status);

            _timer.Remember();
            status = queue.enqueueNDRangeKernel(*kernel, cl::NullRange, globalWorkSize, localWorkSize, nullptr, &event);
            WAIT_AND_CHECK(event, status);
            totalTimeCPU += _timer.Diff();

            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
            totalTimeGPU += (endTime - startTime);
        }
    }

    cout << "CPU: " << totalTimeCPU << ", GPU: " << totalTimeGPU << endl;
}

void Gemm::Cleanup() {
    _deviceMatrixA.reset();
    _deviceMatrixB.reset();
    _deviceMatrixC.reset();
    _sourceMatrixA.reset(),
    _sourceMatrixB.reset();
    _sourceMatrixC.reset();
    _nnKernel.reset();
    _ntKernel.reset();
    _program.reset();
}

template <typename TItem>
void Gemm::RunInternal() {
    if (InitContext<TItem>() == 0) {
        SetKernelArguments<TItem>();
        InitData<TItem>();
        ExecuteKernels();
        Cleanup();
    }
}

void Gemm::Run() {
    cout << "Running: gemm<float>" << endl;
    RunInternal<float>();

    if (_controller->SupportsDoublePrecision()) {
        cout << "Running: gemm<double>" << endl;
        RunInternal<double>();
    }

    cout << "Running: gemm<cl_int>" << endl;
    RunInternal<cl_int>();

    cout << "Running: gemm<cl_long>" << endl;
    RunInternal<cl_long>();

    cout << endl;
}

