#include "fft.hpp"

#include <iostream>
#include <random>
#include <vector>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

// totalBufferSize must be devidable by 512 * sizeof(ComplexValue<TItem>) * 2
static const int TOTAL_BUFFER_SIZE = 256 * 1024 * 1024;
static const int PASSES = 10;

template <typename TItem>
struct ComplexValue {
    TItem x;
    TItem y;
};

template <typename TItem>
int Fft::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    //compilerParams += " -cl-fast-relaxed-math"; do not enable this by default
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "fft.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = 0;
    _checkKernel = make_shared<cl::Kernel>(*_program, "chk1D_512", &status);
    CHECK_RETURN_ERROR(status);
    _forwardKernel = make_shared<cl::Kernel>(*_program, "fft1D_512", &status);
    CHECK_RETURN_ERROR(status);
    _inverseKernel = make_shared<cl::Kernel>(*_program, "ifft1D_512", &status);
    CHECK_RETURN_ERROR(status);

    return 0;
}

template <typename TItem>
void Fft::InitData() {
    int BUFFER_SIZE = TOTAL_BUFFER_SIZE;
    if (sizeof(TItem) == 8)
        BUFFER_SIZE *= 2; // we want the same number of elements in both cases -> float vs double!

    // 512 values in a block, every block is stored twice -> once for processing, once for validation
    _blockSize = 512;
    _blocksToProcessHalf = BUFFER_SIZE / (_blockSize * sizeof(ComplexValue<TItem>) * 2);
    _blocksToProcess = _blocksToProcessHalf * 2;
    _numberOfFFTValuesHalf = _blocksToProcessHalf * _blockSize;

    // allocate buffer on host for input data
    vector<ComplexValue<TItem>> source;
    source.resize(_numberOfFFTValuesHalf * 2);

    // init source memory on host
    random_device randomDevice;
    default_random_engine engine(randomDevice());
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < _numberOfFFTValuesHalf; ++i) {
        source[i].x = static_cast<TItem>(dist(engine) * 2 - 1);      // the values we process
        source[i].y = static_cast<TItem>(dist(engine) * 2 - 1);
        source[i + _numberOfFFTValuesHalf].x = source[i].x;    // backup for validation (?)
        source[i + _numberOfFFTValuesHalf].y = source[i].y;
        // originally I thought these values are used as reference for the check kernel, but fft implementation from SHOC transforms all values
        // therefore to stay consistent we do the same thing...
        // (see FFT.cpp in SHOC where they pass n_ffts instead of half_n_ffts to the transform function)
        // --> do not use the fft.cl implementation for productive code, not sure if the result is correct
    }

    // allocate device memory
    _processBufferDevice = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, BUFFER_SIZE);
    _validationBufferDevice = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(cl_int));

    // copy data to device
    cl::CommandQueue& queue = _controller->Queue();
    cl_int defaultValidationResult = 0;
    queue.enqueueWriteBuffer(*_validationBufferDevice, CL_TRUE, 0, sizeof(defaultValidationResult), &defaultValidationResult); // set check flag to zero
    queue.enqueueWriteBuffer(*_processBufferDevice, CL_TRUE, 0, BUFFER_SIZE, &source[0]); // copy source data
    queue.finish();

    // set kernel arguments
    _forwardKernel->setArg(0, *_processBufferDevice);
    _inverseKernel->setArg(0, *_processBufferDevice);
    _checkKernel->setArg(0, *_processBufferDevice);
    _checkKernel->setArg(1, _numberOfFFTValuesHalf);
    _checkKernel->setArg(2, *_validationBufferDevice);
}

void Fft::ExecuteKernels() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::NDRange localWorkSize(64);
    cl::NDRange globalWorkSizeTransform(64 * _blocksToProcess);
    cl::NDRange globalWorkSizeCheck(64 * _blocksToProcessHalf);
    cl_long endTime, startTime;
    cl_int status = CL_SUCCESS;
    cl::Event event;

    int64_t totalTimeCPU = 0;
    int64_t totalTimeGPU = 0;

    for (int i = 0; i < PASSES; ++i) {
        // forward
        _timer.Remember();
        status = queue.enqueueNDRangeKernel(*_forwardKernel, cl::NullRange, globalWorkSizeTransform, localWorkSize, nullptr, &event);
        WAIT_AND_CHECK(event, status);
        totalTimeCPU += _timer.Diff();

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        totalTimeGPU += (endTime - startTime);

        // inverse
        _timer.Remember();
        status = queue.enqueueNDRangeKernel(*_inverseKernel, cl::NullRange, globalWorkSizeTransform, localWorkSize, nullptr, &event);
        WAIT_AND_CHECK(event, status);
        totalTimeCPU += _timer.Diff();

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        totalTimeGPU += (endTime - startTime);

        // check
        cl_int result = 0;
        status = queue.enqueueNDRangeKernel(*_checkKernel, cl::NullRange, globalWorkSizeCheck, localWorkSize, nullptr, &event);
        WAIT_AND_CHECK(event, status);

        status = queue.enqueueReadBuffer(*_validationBufferDevice, CL_TRUE, 0, sizeof(result), &result, nullptr, &event);
        WAIT_AND_CHECK(event, status);

        if (result != 0) {
            cerr << "FFT/IFFT result invalid" << endl;
        }
    }

    cout << "CPU: " << totalTimeCPU << ", GPU: " << totalTimeGPU << endl;
}

void Fft::Cleanup() {
    _processBufferDevice.reset();
    _validationBufferDevice.reset();
    _checkKernel.reset();
    _forwardKernel.reset();
    _inverseKernel.reset();
    _program.reset();
}

template <typename TItem>
void Fft::RunInternal() {
    if (InitContext<TItem>() == 0) {
        InitData<TItem>();
        ExecuteKernels();
        Cleanup();
    }
}

void Fft::Run() {
    cout << "Running fft<float>" << endl;
    RunInternal<float>();

    if (_controller->SupportsDoublePrecision()) {    
        cout << "Running fft<double>" << endl;
        RunInternal<double>();
    }

    cout << endl;
}

