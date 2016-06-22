#include "applicationcontroller.hpp"

#include "benchmarks/api.hpp"
#include "benchmarks/blackscholes.hpp"
#include "benchmarks/cfd.hpp"
#include "benchmarks/edge.hpp"
#include "benchmarks/fft.hpp"
#include "benchmarks/gemm.hpp"
#include "benchmarks/kmeans.hpp"
#include "benchmarks/memory.hpp"
#include "benchmarks/spmv.hpp"
#include "benchmarks/stencil.hpp"
#include "benchmarks/streamcluster.hpp"
#include "benchmarks/transpose.hpp"
#include "benchmarks/vecop.hpp"

#include <iostream>

using namespace std;

static const char* HELP_TEXT = "OpenCL Benchmark-Collection\n"
            "Author: Michael Eiler <eiler.mike@gmail.com>\n\n"
            "  --run-<benchmark> executes only the selected benchmarks, available benchmarks are:\n\n"
            "    api, blackscholes, cfd, edge, fft, gemm, kmeans, memory, spmv, stencil, streamcluster, transpose, vecop\n\n"
            "  --opt-disable disable all optimizations (-cl-mad-enable is passed to the compiler by default)\n"
            "  --opt-speed enables additional otimizations (-cl-fast-relaxed-math and -cl-no-signed-zeros)\n"
            "  --save-binaries stores all compiled cl-files (programs) in the execution directory\n"
            "  --verbose / -v prints more platform and device information\n"
            "  --help / -h prints this information\n";

ApplicationController::ApplicationController()
    : _tests()
    , _runSpecificTests() {

}

ApplicationController::~ApplicationController() {

}

template <typename TClass>
void ApplicationController::CreateTestInstance(const std::string& name) {
    bool createInstance = _runSpecificTests.size() == 0;

    if (!createInstance) {
        for (const auto& testName : _runSpecificTests) {
            if (testName == name) {
                createInstance = true;
                break;
            }
        }
    }

    if (createInstance) {
        auto test = make_shared<TClass>(_controller);
        test->RequestDisableOptimization(_disableOptimization);
        test->RequestOptimizationForSpeed(_optimizeForSpeed);
        _tests.push_back(test);
    }
}

void ApplicationController::CreateAndExecuteTests() {
    CreateTestInstance<benchmarks::Api>("api");
    CreateTestInstance<benchmarks::BlackScholes>("blackscholes");
    CreateTestInstance<benchmarks::Cfd>("cfd");
    CreateTestInstance<benchmarks::Edge>("edge");
    CreateTestInstance<benchmarks::Fft>("fft");
    CreateTestInstance<benchmarks::Gemm>("gemm");
    CreateTestInstance<benchmarks::KMeans>("kmeans");
    CreateTestInstance<benchmarks::Memory>("memory");
    CreateTestInstance<benchmarks::Spmv>("spmv");
    CreateTestInstance<benchmarks::Stencil>("stencil");
    CreateTestInstance<benchmarks::StreamCluster>("streamcluster");
    CreateTestInstance<benchmarks::Transpose>("transpose");
    CreateTestInstance<benchmarks::Vecop>("vecop");

    for (auto& test : _tests) {
        test->Run();
        test.reset();
    }
}

int ApplicationController::Run(int argc, char** argv) {
    bool saveBinaries = false;
    bool showDetails = false;

    for (int i = 1; i < argc; ++i) {
        string argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            cout << HELP_TEXT;
            return 1;
        }
        if (argument == "--opt-disable") {
            _disableOptimization = true;
        }
        if (argument == "--opt-speed") {
            _optimizeForSpeed = true;
        }
        if (argument == "--save-binaries") {
            saveBinaries = true;
        }
        if (argument == "--verbose" || argument == "-v") {
            showDetails = true;
        }
        if (argument.find("--run-") == 0) {
            _runSpecificTests.push_back(argument.substr(6));
        }
    }

    _controller = make_shared<ComputeController>();
    _controller->SetSaveProgramBinaries(saveBinaries);

    int status = _controller->SelectDeviceDialog(showDetails);
    if (status == 0)
        CreateAndExecuteTests();

    return status;
}

