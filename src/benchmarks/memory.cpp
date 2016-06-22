#include "memory.hpp"

#include <iostream>
#include <memory.h>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int BUFFER_SIZE = 256 * 1024 * 1024; // 256 MiB
static const int ITERATIONS = 100; // number of iterations

int Memory::AlignmentFactor() {
    cl_uint factor = -1;

    _controller->SelectedDevice().getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &factor);

    return factor;
}

template <typename TItem>
TItem* Memory::AlignAddress(TItem* address, const int alignment, const int displacement) {
    uint64_t startAddress = reinterpret_cast<uint64_t>(address);

    if ((alignment & (alignment - 1)) != 0) // return nullptr if alignment is not a power of two
        return nullptr;

    startAddress &= ~(alignment - 1);   // convert ...0000100...  to 000011... and then to 111100...
                                        // apply bitmask to calculate an aligned address
    startAddress += alignment;          // readd alignment so we are within the memory we allocated
    startAddress += displacement;       // add displacement in case we deffinitely want unaligned memory

    return reinterpret_cast<TItem*>(startAddress);
}

template<typename TItem>
void Memory::FillBufferWithContent(TItem *buffer, int length) {
    for (int i = 0; i < length; ++i) {
        buffer[i] = (TItem)i;
    }
}

// is this memory really pinned after passing it to cl::Buffer???
void Memory::CopyMemoryToDevice(bool align) {
    int length = BUFFER_SIZE / sizeof(float);

    // allocate buffer using correct assignment values
    int displacement = (align ? 0 : 83);
    int alignmentFactor = AlignmentFactor();
    void *buffer = malloc(BUFFER_SIZE + alignmentFactor + displacement);
    float *hostBuffer = AlignAddress<float>(static_cast<float*>(buffer), alignmentFactor, displacement);

    string testName = "CopyMemoryToDevice ";
    testName += (align ? "(aligned),  " : "(unaligned),");

    cl::CommandQueue& queue = _controller->Queue();
    FillBufferWithContent<float>(hostBuffer, length);
    cl::Buffer pinnedMemoryBuffer(_controller->Context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, static_cast<size_t>(BUFFER_SIZE), static_cast<void*>(hostBuffer), nullptr);

    cl::Buffer targetBuffer(_controller->Context(), CL_MEM_READ_WRITE, BUFFER_SIZE);

    // actual test function
    PerformTest([&](cl::Event& event) -> void { 
            queue.enqueueCopyBuffer(pinnedMemoryBuffer, targetBuffer, 0, 0, BUFFER_SIZE, nullptr, &event);
        }, testName, ITERATIONS);

    free(buffer);
}

void Memory::CopyUnpinnedMemoryToDevice() {
    cl::CommandQueue& queue = _controller->Queue();

    int alignmentFactor = AlignmentFactor();
    void *buffer = malloc(BUFFER_SIZE + alignmentFactor);
    float *hostBuffer = AlignAddress<float>(static_cast<float*>(buffer), alignmentFactor, 0);

    int length = BUFFER_SIZE / sizeof(float);
    FillBufferWithContent<float>(hostBuffer, length);

    cl::Buffer targetBuffer(_controller->Context(), CL_MEM_READ_WRITE, BUFFER_SIZE);

    string testName = "CopyUnpinnedMemoryToDevice,    ";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueWriteBuffer(targetBuffer, CL_TRUE, 0, BUFFER_SIZE, static_cast<void*>(hostBuffer), nullptr, &event);
        }, testName, ITERATIONS);

    free(buffer);
}

void Memory::CopyToHostMemory() {
    cl::CommandQueue& queue = _controller->Queue();

    int alignmentFactor = AlignmentFactor();
    void *buffer = malloc(BUFFER_SIZE + alignmentFactor);
    float *positionedBuffer = AlignAddress<float>(static_cast<float*>(buffer), alignmentFactor, 0);

    int length = BUFFER_SIZE / sizeof(float);
    FillBufferWithContent<float>(positionedBuffer, length);

    cl::Buffer deviceBuffer(_controller->Context(), CL_MEM_READ_WRITE, BUFFER_SIZE);
    queue.enqueueWriteBuffer(deviceBuffer, CL_TRUE, 0, BUFFER_SIZE, static_cast<void*>(positionedBuffer));
    queue.finish();

    cl::Buffer hostBuffer(_controller->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BUFFER_SIZE);

    string testName = "CopyToHostMemory,              ";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueCopyBuffer(deviceBuffer, hostBuffer, 0, 0, BUFFER_SIZE, nullptr, &event);
        }, testName, ITERATIONS);

    free(buffer);
}

void Memory::CopyToUnpinnedHostMemory() {
    cl::CommandQueue& queue = _controller->Queue();

    int alignmentFactor = AlignmentFactor();
    void *buffer = malloc(BUFFER_SIZE + alignmentFactor);
    float *positionedBuffer = AlignAddress<float>(static_cast<float*>(buffer), alignmentFactor, 0);

    int length = BUFFER_SIZE / sizeof(float);
    FillBufferWithContent<float>(positionedBuffer, length);

    cl::Buffer deviceBuffer(_controller->Context(), CL_MEM_READ_WRITE, BUFFER_SIZE);
    queue.enqueueWriteBuffer(deviceBuffer, CL_TRUE, 0, BUFFER_SIZE, static_cast<void*>(positionedBuffer));
    queue.finish();

    cl::Buffer hostBuffer(_controller->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BUFFER_SIZE);

    string testName = "CopyToUnpinnedHostMemory,      ";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueReadBuffer(deviceBuffer, CL_TRUE, 0, BUFFER_SIZE, positionedBuffer, nullptr, &event);
        }, testName, ITERATIONS);

    free(buffer);
}

void Memory::WriteToHostMemory(bool align) {
    string compilerParams = GetCompilerFlags<float>();
    auto program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "memory.cl", compilerParams);

    cl_int status = 0;
    cl::Kernel writeToHostKernel(*program, "write_to_buffer_loop_simple", &status);
    CHECK(status);

    cl::CommandQueue& queue = _controller->Queue();

    // allocate buffer using correct assignment values
    int displacement = (align ? 0 : 83);
    int alignmentFactor = AlignmentFactor();
    void *buffer = malloc(BUFFER_SIZE + alignmentFactor + displacement);
    float *repositionedBuffer = AlignAddress<float>(static_cast<float*>(buffer), alignmentFactor, displacement);
    cl::Buffer hostBuffer(_controller->Context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, static_cast<size_t>(BUFFER_SIZE), repositionedBuffer, nullptr);

    // initialize kernel
    cl_int blockSizeCl = 128;// 4;
    cl_int lengthCl = BUFFER_SIZE / sizeof(float);
    cl::NDRange global(lengthCl / blockSizeCl);
    writeToHostKernel.setArg(0, hostBuffer);
    writeToHostKernel.setArg(1, blockSizeCl);
    writeToHostKernel.setArg(2, lengthCl);

    string testName = "WriteToHostMemory ";
    testName += (align ? "(aligned),   " : "(unaligned), ");

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueNDRangeKernel(writeToHostKernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
        }, testName, ITERATIONS);

    free(buffer);
}

void Memory::ReadFromHostMemory() {
    string compilerParams = GetCompilerFlags<float>();
    auto program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "memory.cl", compilerParams);

    cl_int status = 0;
    cl::Kernel readKernel(*program, "read_from_buffer_vectorized", &status);
    CHECK(status);

    int length = BUFFER_SIZE / sizeof(float);
    float *buffer = new float[length];
    FillBufferWithContent<float>(buffer, length);

    cl_int blockSizeCl = 128;// 4;
    cl::Buffer deviceBuffer(_controller->Context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, static_cast<size_t>(BUFFER_SIZE), buffer, nullptr);
    cl::Buffer targetBuffer(_controller->Context(), CL_MEM_READ_WRITE, sizeof(float) * BUFFER_SIZE / blockSizeCl);

    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(deviceBuffer, CL_TRUE, 0, BUFFER_SIZE, static_cast<void*>(buffer));
    queue.finish();

    cl::NDRange global(length / blockSizeCl);
    readKernel.setArg(0, deviceBuffer);
    readKernel.setArg(1, targetBuffer);
    readKernel.setArg(2, blockSizeCl);
    readKernel.setArg(3, length);

    string testName = "ReadFromHostMemory,            ";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueNDRangeKernel(readKernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
        }, testName, ITERATIONS);

    delete[] buffer;
}

void Memory::AnalyseMemoryAccessPatterns() {
    string compilerParams = GetCompilerFlags<float>();
    auto program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "memory.cl", compilerParams);
    cl::CommandQueue& queue = _controller->Queue();

    cl_int status = 0;
    cl::Kernel kernelRM(*program, "read_from_buffer_row_major", &status);
    CHECK(status);
    cl::Kernel kernelCM(*program, "read_from_buffer_column_major", &status);
    CHECK(status);

    const int numberOfRows = 4096; // 4096 * 4096 * sizeof(float) = 64MB
    const int numberOfColumns = 4096;   // numberOfRows will represent the number of global work size

    vector<float> matrixRM, matrixCM;
    matrixRM.resize(numberOfRows * numberOfColumns);
    matrixCM.resize(numberOfRows * numberOfColumns);
    for (int i = 0; i < numberOfRows; ++i) {
        for (int j = 0; j < numberOfColumns; ++j) {
            float value = static_cast<float>(i * numberOfRows + j) / static_cast<float>(numberOfRows * numberOfColumns);
            matrixRM[i * numberOfRows + j] = value;
            matrixCM[j * numberOfColumns + i] = value;
        }
    }

    const int bufferSize = sizeof(float) * numberOfRows * numberOfColumns;
    cl::Buffer inputBufferRM(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    cl::Buffer inputBufferCM(_controller->Context(), CL_MEM_READ_WRITE, bufferSize);
    queue.enqueueWriteBuffer(inputBufferRM, CL_TRUE, 0, bufferSize, &matrixRM[0]);
    queue.enqueueWriteBuffer(inputBufferCM, CL_TRUE, 0, bufferSize, &matrixCM[0]);
    queue.finish();

    cl::Buffer outputBuffer(_controller->Context(), CL_MEM_READ_WRITE, numberOfRows * sizeof(float));

    kernelRM.setArg(0, inputBufferRM);
    kernelRM.setArg(1, outputBuffer);
    kernelRM.setArg(2, numberOfRows);
    kernelRM.setArg(3, numberOfColumns);

    kernelCM.setArg(0, inputBufferCM);
    kernelCM.setArg(1, outputBuffer);
    kernelCM.setArg(2, numberOfRows);
    kernelCM.setArg(3, numberOfColumns);

    vector<float> resultRM, resultCM;
    resultRM.resize(numberOfRows);
    resultCM.resize(numberOfRows);

    cl::NDRange global(numberOfRows);
    string testName = "MemoryAccessPatterns, RowMajor,";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueNDRangeKernel(kernelRM, cl::NullRange, global, cl::NullRange, nullptr, &event);
        }, testName, ITERATIONS);

    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * numberOfRows, &resultRM[0]);
    queue.finish();

    testName = "MemoryAccessPatterns, ColMajor,";

    PerformTest([&](cl::Event& event) -> void {
            queue.enqueueNDRangeKernel(kernelCM, cl::NullRange, global, cl::NullRange, nullptr, &event);
        }, testName, ITERATIONS);

    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * numberOfRows, &resultCM[0]);
    queue.finish();

    for (int i = 0; i < numberOfRows; ++i) {
        if (resultRM[i] != resultCM[i]) {
            cout << "result not equal" << endl;
            return;
        }
    }
}

void Memory::Run() {
    cout << "Memory Benchmarks" << endl;

    CopyMemoryToDevice(true);
    CopyMemoryToDevice(false);
    CopyUnpinnedMemoryToDevice();
    CopyToHostMemory();
    CopyToUnpinnedHostMemory();
    WriteToHostMemory(true);
    WriteToHostMemory(false);

    int workGroupSizes[] = { 32, 64, 128, 192, 256, 512 };

    for (auto& workGroupSize : workGroupSizes) {
        RequestWorkGroupSize(workGroupSize);

        cout << "WorkGroupSize set to: " << workGroupSize << endl;

        ReadFromHostMemory();
        AnalyseMemoryAccessPatterns();
    }

    cout << endl;
}
