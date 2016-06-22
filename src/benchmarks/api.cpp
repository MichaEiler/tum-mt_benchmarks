#include "api.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"
#include "../statistics.hpp"

using namespace benchmarks;
using namespace std;

static const int COMPILE_ITERATIONS = 10;

Api::Api(shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

Api::~Api() {

}

bool LoadFile(const string&, string&);

void Api::CompilationTimeAnalysis(const string& kernelSourceFile) {
    cout << "CompilationTimeAnalysis(\"" << kernelSourceFile << "\"): " << endl;

    string code = "";
    if (!LoadFile(CL_SRC_PATH_PREFIX + kernelSourceFile, code)) {
        cerr << "Could not find source file " << kernelSourceFile << endl;
        return;
    }

    Statistics<int64_t> stats;

    auto compilerParams = GetCompilerFlags<float>();
    shared_ptr<cl::Program> program;
    for (int i = 0; i < COMPILE_ITERATIONS; ++i) {
        _timer.Remember();
        program = _controller->BuildFromSourceStr(code, compilerParams);
        stats.Add(_timer.Diff());
        if (program.get() == nullptr) {
            cerr << "Failed to compile program from source." << endl;
            return;
        }
    }

    cout << "Compilation from source, avg: " << stats.Mean() << ", deviation: " << stats.Deviation<int64_t>()
         << ", max: " << stats.Max() << ", min: " << stats.Min() << endl;

    stats.Clear();

    vector<char> binary;
    _controller->SavePlatformSpecificBinary(program, binary);

    for (int i = 0; i < COMPILE_ITERATIONS; ++i) {
        _timer.Remember();
        program = _controller->BuildFromBinary(binary, compilerParams);
        stats.Add(_timer.Diff());
        if (program.get() == nullptr) {
            cerr << "Failed to compile program from binary." << endl;
            return;
        }
    }

    cout << "Compilation from binary, avg: " << stats.Mean() << ", deviation: " << stats.Deviation<int64_t>()
         << ", max: " << stats.Max() << ", min: " << stats.Min() << endl;
}

void Api::Run() {
    auto kernelFiles =  { 
                            "blackscholes.cl", "cfd.cl", "fft.cl", "gemm.cl", 
                            "kmeans.cl", "spmv.cl", "streamcluster.cl", "transpose.cl"
                        };

    for (const auto& kernelFile : kernelFiles) {
        CompilationTimeAnalysis(kernelFile);
    }

    cout << endl;
}

bool LoadFile(const string& path, string& buffer) {
    ifstream stream(path, ios::in);
    if (stream) {
        stream.seekg(0, ios::end);
        buffer.resize(static_cast<size_t>(stream.tellg()));
        stream.seekg(0, ios::beg);
        stream.read(static_cast<char*>(&buffer[0]), buffer.size());
        stream.close();
        return true;
    }

    return false;
}
