#include "benchmarkbase.hpp"

#include "clglobal.hpp"
#include "computecontroller.hpp"

#include <iostream>

using namespace benchmarks;
using namespace std;

BenchmarkBase::BenchmarkBase(std::shared_ptr<ComputeController> controller)
    : _controller(controller)
    , _timer()
    , _cpuStatistics()
    , _gpuStatistics()
    , _requestedWorkGroupSize(-1)
    , _optimizeForSpeed(false)
    , _disableOptimization(false) {

}

BenchmarkBase::~BenchmarkBase() {

}

string BenchmarkBase::GetCompilerFlagsInternal(const type_info& typeInfo) {
    string params = "";

    if (typeInfo == typeid(float)) {
        params = "-DVTYPE_FLOAT";
    } else if (typeInfo == typeid(double)) {
        if (_controller->HasExtension("cl_khr_fp64"))
            params += "-DVTYPE_DOUBLE_KHR";
        else if (_controller->HasExtension("cl_amd_fp64"))
            params += "-DVTYPE_DOUBLE_AMD";
    }  else if (typeInfo == typeid(cl_long)) {
        params += "-DVTYPE_LONG";
    } else if (typeInfo == typeid(cl_int)) {
        params += "-DVTYPE_INT";
    }

    if (_disableOptimization)
        return params + " -cl-opt-disable";

    params += " -cl-mad-enable"; // replace a*b+c with MAD instruction
    if (_optimizeForSpeed)
        params += " -cl-fast-relaxed-math -cl-no-signed-zeros";
//    cout << params << endl;

    return params;
}

void BenchmarkBase::PerformTest(function<void(cl::Event&)> testFunction, const string& testName, const int iterations){
    _cpuStatistics.Clear();
    _gpuStatistics.Clear();

    for (int i = 0; i < iterations; ++i) {
        cl::Event event;
        cl_ulong startTime, endTime;

        _timer.Remember();

        testFunction(event);

        event.wait();
        _cpuStatistics.Add(_timer.Diff());

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        _gpuStatistics.Add(endTime - startTime);
    }

    cout << testName << " CPU: " << _cpuStatistics.Mean() << " (+/- " << _cpuStatistics.Deviation<int64_t>() 
        << "), GPU: " << _gpuStatistics.Mean() << " (+/- " << _gpuStatistics.Deviation<int64_t>() << ")" << endl;
}

int BenchmarkBase::RoundToPowerOf2(int i, int powerOf2) {
    int bitmask = powerOf2 - 1;  // 001000 -> 000111
    int remaining = i & bitmask;
    return i + (powerOf2 - remaining);
}

int BenchmarkBase::RoundToMultipleOf(int i, int p) {
    return i + ((p - (i%p)) % p);
}

void BenchmarkBase::RequestWorkGroupSize(int workGroupSize) {
    _requestedWorkGroupSize = workGroupSize;
}
