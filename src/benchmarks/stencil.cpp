#include "stencil.hpp"

#include <iostream>
#include <random>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int TEST_ITERATIONS = 10;
static const int ALGORITHM_ITERATIONS = 1000;
static const int MATRIX_HEIGHT = 1024;
static const int MATRIX_WIDTH = 1024;
static const float WEIGHT_CARDINAL = 0.15f;
static const float WEIGHT_CENTER = 0.25f;
static const float WEIGHT_DIAGONAL = 0.05f;
static const int LOCAL_ROWS = 8;
static const int LOCAL_COLUMNS = 256;

Stencil::Stencil(std::shared_ptr<ComputeController> controller) : BenchmarkBase(controller) {

}

Stencil::~Stencil() {

}

template <typename TItem>
void Stencil::FillMatrix(cl::Buffer &buffer, int width, int height) {
    cl::CommandQueue& queue = _controller->Queue();

    vector<TItem> dataBuffer;
    dataBuffer.resize(width * height);

    for (int i = 0; i < width * height; ++i) {
        dataBuffer[i] = ((float)i) / width;
    }

    const int bufferSize = width * height * sizeof(TItem);
    cl_int status = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, bufferSize, static_cast<void*>(&dataBuffer[0]));
    if (status != CL_SUCCESS) {
        cerr << "stencil: failed to copy input data to device" << endl;
    }
}

template <typename TItem>
int Stencil::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    compilerParams += " -DLOCAL_ROWS=" + to_string(LOCAL_ROWS);
    compilerParams += " -DLOCAL_COLUMNS=" + to_string(LOCAL_COLUMNS);
    compilerParams += " -DGLOBAL_ROWS=" + to_string(MATRIX_HEIGHT);
    compilerParams += " -DGLOBAL_COLUMNS=" + to_string(MATRIX_WIDTH);

    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "stencil2d.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = 0;
    _stencilKernel = make_shared<cl::Kernel>(*_program, "StencilKernel", &status);
    CHECK_RETURN_ERROR(status);

    const int bufferSize = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(TItem);
    _sourceBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _inputBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _outputBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);

    FillMatrix<TItem>(*_sourceBuffer, MATRIX_WIDTH, MATRIX_HEIGHT);

    return 0;
}

template <typename TItem>
void Stencil::RunInternal() {
    if (InitContext<TItem>() != 0)
        return;

    cl::CommandQueue& queue = _controller->Queue();

    cl_int alignment = 16;
    TItem center = WEIGHT_CENTER, cardinal = WEIGHT_CARDINAL, diagonal = WEIGHT_DIAGONAL;

    cl::NDRange localWorkSize(1, LOCAL_COLUMNS);    // 1 means the kernel handles eight rows (LOCAL_ROWS compiler flag)
    cl::NDRange globalWorkSize((MATRIX_HEIGHT - 2) / LOCAL_ROWS,
        RoundToPowerOf2(MATRIX_WIDTH - 2, LOCAL_COLUMNS));
    cl_long startTime, endTime;
    cl::Event event;

    int64_t totalTimeCPU = 0;
    int64_t totalTimeGPU = 0;

    // set kernel arguments
    _stencilKernel->setArg(2, alignment);
    _stencilKernel->setArg(3, center);
    _stencilKernel->setArg(4, cardinal);
    _stencilKernel->setArg(5, diagonal);

    for (int i = 0; i < TEST_ITERATIONS; ++i) {

        auto currentBuffer = _inputBuffer;
        auto otherBuffer = _outputBuffer;

        // initialize/restore original input buffer
        cl_int status = queue.enqueueCopyBuffer(*_sourceBuffer, *_inputBuffer, 0, 0, MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(TItem), nullptr, &event);
        WAIT_AND_CHECK(event, status);

        for (int j = 0; j < ALGORITHM_ITERATIONS; ++j) {
            // update kernel arguments
            _stencilKernel->setArg(0, *currentBuffer);
            _stencilKernel->setArg(1, *otherBuffer);

            // execute kernel
            _timer.Remember();
            status = queue.enqueueNDRangeKernel(*_stencilKernel, cl::NullRange, globalWorkSize, localWorkSize, nullptr, &event);
            WAIT_AND_CHECK(event, status);
            totalTimeCPU += _timer.Diff();

            queue.finish();

            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
            totalTimeGPU += (endTime - startTime);

            // swap buffers, output is new input for next interation
            auto temporaryBuffer = currentBuffer;
            currentBuffer = otherBuffer;
            otherBuffer = temporaryBuffer;
        }
    }

    totalTimeCPU /= TEST_ITERATIONS;
    totalTimeGPU /= TEST_ITERATIONS;

    cout << " CPU: " << totalTimeCPU << ", GPU: " << totalTimeGPU << endl;
}

void Stencil::Run() {
    cout << "Stencil2D Test: " << endl;

    cout << "Stencil<float>";
    RunInternal<float>();
    if (_controller->SupportsDoublePrecision()) {
        cout << "Stencil<double>";
        RunInternal<double>();
    }

    cout << endl;
}
