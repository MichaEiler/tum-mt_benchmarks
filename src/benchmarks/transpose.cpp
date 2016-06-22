#include "transpose.hpp"

#include <iostream>
#include <memory>
#include <random>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

// 8192 * 4096 * sizeof(float) = 128 MiB
static const int MATRIX_WIDTH = 512;// 4096;
static const int MATRIX_HEIGHT = 256;// 2048;
static const int ITERATIONS = 100;

static const int BLOCK_DIMENSION = 16; // if you change, also update the define in transpose.cl

Transpose::Transpose(std::shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

Transpose::~Transpose() {

}

template <typename TItem>
void Transpose::GenerateInputData(int width, int height, TItem *buffer) {
    for (int i = 0; i < width * height; ++i) {
        buffer[i] = static_cast<TItem>(i);
    }
}

int Transpose::InitContext() {
    // compile program
    string compilerParams = GetCompilerFlags<float>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "transpose.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = 0;
    _transposeSimpleKernel = make_shared<cl::Kernel>(*_program, "transpose_simple", &status);
    CHECK_RETURN_ERROR(status);
    _transposeOptimizedKernel = make_shared<cl::Kernel>(*_program, "transpose_optimized", &status);
    CHECK_RETURN_ERROR(status);

    // initialize data structures
    const int bufferSize = MATRIX_HEIGHT * MATRIX_WIDTH * sizeof(float);
    _inputBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _outputBuffer1 = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    _outputBuffer2 = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);

    vector<float> buffer;
    buffer.resize(MATRIX_HEIGHT * MATRIX_WIDTH);
    GenerateInputData(MATRIX_WIDTH, MATRIX_HEIGHT, &buffer[0]);
    _controller->Queue().enqueueWriteBuffer(*_inputBuffer, CL_TRUE, 0, bufferSize, static_cast<void*>(&buffer[0]));

    // set kernel parameters
    cl_int width = MATRIX_WIDTH;
    cl_int height = MATRIX_HEIGHT;

    _transposeSimpleKernel->setArg(0, *_inputBuffer);
    _transposeSimpleKernel->setArg(1, *_outputBuffer1);
    _transposeSimpleKernel->setArg(2, height);
    _transposeSimpleKernel->setArg(3, width);

    _transposeOptimizedKernel->setArg(0, *_inputBuffer);
    _transposeOptimizedKernel->setArg(1, *_outputBuffer2);
    _transposeOptimizedKernel->setArg(2, height);
    _transposeOptimizedKernel->setArg(3, width);

    return 0;
}

void Transpose::Validate() {
    // compare if output buffers are equal
    const int items = MATRIX_HEIGHT * MATRIX_WIDTH;
    const int bufferSize = items * sizeof(float);

    vector<float> bufferSimple, bufferOptimized;
    bufferSimple.resize(items);
    bufferOptimized.resize(items);

    cl::CommandQueue &queue = _controller->Queue();
    queue.enqueueReadBuffer(*_outputBuffer1, CL_TRUE, 0, bufferSize, &bufferSimple[0]);
    queue.enqueueReadBuffer(*_outputBuffer2, CL_TRUE, 0, bufferSize, &bufferOptimized[0]);
    queue.finish();

    for (int i = 0; i < items; ++i) {
        if (bufferSimple[i] != bufferOptimized[i]) {
            cerr << "Buffers are different at item " << i << ", bufferSimple[i]: " << bufferSimple[i]
                << ", bufferOptimized[i]: " << bufferOptimized[i] << endl;
            break;
        }
    }
}

void Transpose::RunSimple() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::Event event;

    cl::NDRange global(MATRIX_WIDTH * MATRIX_HEIGHT);

    string testName = "Transpose::RunSimple";

    PerformTest([&](cl::Event &event) -> void {
            queue.enqueueNDRangeKernel(*_transposeSimpleKernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
        }, testName, ITERATIONS);
}

void Transpose::RunOptimized() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::Event event;

    cl::NDRange local(16, 16);
    cl::NDRange global(MATRIX_WIDTH, MATRIX_HEIGHT);

    string testName = "Transpose::RunOptimized";

    PerformTest([&](cl::Event &event) -> void {
            queue.enqueueNDRangeKernel(*_transposeOptimizedKernel, cl::NullRange, global, local, nullptr, &event);
        }, testName, ITERATIONS);
}


void Transpose::Run() {
    cout << "Transpose-Tests: " << endl;

    if (InitContext() == 0) {
        RunSimple();
        RunOptimized();
        Validate();
    }

    cout << endl;
}