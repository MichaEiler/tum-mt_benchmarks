#include "edge.hpp"

#include <iostream>
#include <random>
#include <vector>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const bool USE_STATIC_IMAGE = false;
static const int WIDTH = 8192;
static const int HEIGHT = 8192;
static const int RANDOM_SEED = 85733;
static const int TEST_ITERATIONS = 50;


Edge::Edge(std::shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

Edge::~Edge() {

}

int Edge::InitContext() {
    string compilerParams = GetCompilerFlags<float>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "edge.cl", compilerParams);

    cl_int status = CL_SUCCESS;
    _edgeKernel = make_shared<cl::Kernel>(*_program, "find_edge_pixels", &status);
    CHECK_RETURN_ERROR(status);
    _optimizedEdgeKernel = make_shared<cl::Kernel>(*_program, "find_edge_pixels_optimized", &status);
    CHECK_RETURN_ERROR(status);

    _imageBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, WIDTH * HEIGHT * sizeof(float));
    _edgeSteepnessBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, (WIDTH - 2) * (HEIGHT - 2) * sizeof(int));

    _edgeKernel->setArg(0, *_imageBuffer);
    _edgeKernel->setArg(1, *_edgeSteepnessBuffer);
    _edgeKernel->setArg(2, WIDTH);
    _edgeKernel->setArg(3, HEIGHT);

    _optimizedEdgeKernel->setArg(0, *_imageBuffer);
    _optimizedEdgeKernel->setArg(1, *_edgeSteepnessBuffer);
    _optimizedEdgeKernel->setArg(2, WIDTH);
    _optimizedEdgeKernel->setArg(3, HEIGHT);

    return 0;
}

void Edge::InitData() {
    vector<float> imageData;
    imageData.resize(WIDTH * HEIGHT);

    default_random_engine randomEngine(RANDOM_SEED);
    uniform_real_distribution<float> valueDistribution(0.0, 100.0);

    for (size_t i = 0; i < imageData.size(); ++i) {
        imageData[i] = USE_STATIC_IMAGE ? 1.0f : valueDistribution(randomEngine);
    }

    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_imageBuffer, CL_TRUE, 0, sizeof(float) * imageData.size(), &imageData[0]);
    queue.finish();
}

void Edge::ExecuteKernel() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::NDRange local(_requestedWorkGroupSize);
    cl::NDRange global(RoundToMultipleOf((WIDTH-2)*(HEIGHT-2), _requestedWorkGroupSize));

    string testName = "EdgeDetection: ";

    PerformTest([&](cl::Event& event) -> void {
            cl_int status = queue.enqueueNDRangeKernel(*_edgeKernel, cl::NullRange, global, local, nullptr, &event);
            WAIT_AND_CHECK(event, status);
        }, testName, TEST_ITERATIONS);

    testName = "EdgeDetection (optimized): ";

    PerformTest([&](cl::Event& event) -> void {
            cl_int status = queue.enqueueNDRangeKernel(*_optimizedEdgeKernel, cl::NullRange, global, local, nullptr, &event);
            WAIT_AND_CHECK(event, status);
        }, testName, TEST_ITERATIONS);
}

void Edge::Cleanup() {
    _imageBuffer.reset();
    _edgeSteepnessBuffer.reset();
    _edgeKernel.reset();
    _optimizedEdgeKernel.reset();
    _program.reset();
}


void Edge::Run() {
    if (InitContext() == 0) {
        InitData();
        RequestWorkGroupSize(128);
        ExecuteKernel();
        Cleanup();
    }

    cout << endl;
}