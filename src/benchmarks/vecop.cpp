#include "vecop.hpp"

#include <iostream>
#include <random>
#include <vector>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

template<typename TItem>
void Vecop::RunInternal(const string& vectorOperation) {
    string compilerParams = GetCompilerFlags<TItem>();
    shared_ptr<cl::Program> program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "vecadd.cl", compilerParams);

    if (program.get() == nullptr)
        return;

    cl_int status = 0;
    cl::Kernel vecadd_kernel(*program, vectorOperation.c_str(), &status);
    CHECK(status);
 
    // create input data   
    const int elements = 32768 * 1024; // 8MB/16MB depending on TItem size 4byte or 8byte
    vector<TItem> inputA, inputB, output;
    inputA.resize(elements);
    inputB.resize(elements);
    output.resize(elements);

    for (int i = 0; i < elements; ++i) {
        inputA[i] = inputB[i] = (TItem)(i + 1);
    }

    // allocate buffers on device    
    const int bufferSize = sizeof(TItem) * elements;
    cl::Buffer inputBufferA = cl::Buffer(_controller->Context(), CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer inputBufferB = cl::Buffer(_controller->Context(), CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer outputBufferC = cl::Buffer(_controller->Context(), CL_MEM_WRITE_ONLY, bufferSize);

    // copy data to device
    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(inputBufferA, CL_TRUE, 0, bufferSize, &inputA[0]);
    queue.enqueueWriteBuffer(inputBufferB, CL_TRUE, 0, bufferSize, &inputB[0]);
    
    vecadd_kernel.setArg(0, inputBufferA);
    vecadd_kernel.setArg(1, inputBufferB);
    vecadd_kernel.setArg(2, outputBufferC);

    // run gpu code
    cl::NDRange global(elements);
    cl::NDRange local(256);
    cl::Event event;
    cl_ulong startTime, endTime;

    queue.flush();
    _timer.Remember();
    status = queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, local, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    int64_t timeCPU = _timer.Diff();

    // read result
    queue.enqueueReadBuffer(outputBufferC, CL_TRUE, 0, bufferSize, &output[0]);
    queue.finish();
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);

    cout << "CPU: " << timeCPU << ", GPU: " << (endTime - startTime) << endl;
}

void Vecop::Run() {
    string operations[] { "vecadd", "vecmul", "vecdiv" };

    for (int i = 0; i < 3; ++i) {
        cout << "Running: " << operations[i] << "<float>" << endl;
        RunInternal<float>(operations[i]);

        if (_controller->SupportsDoublePrecision()) {
            cout << "Running: " << operations[i] << "<double>" << endl;
            RunInternal<double>(operations[i]);
        }

        cout << "Running: " << operations[i] << "<cl_int>" << endl;
        RunInternal<cl_int>(operations[i]);
        cout << "Running: "<< operations[i] << "<cl_long>" << endl;
        RunInternal<cl_long>(operations[i]);
    }
    cout << endl;
}

