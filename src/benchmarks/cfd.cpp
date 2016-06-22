#include "cfd.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const string CFD_DATASET = "cfd/fvcorr.domn.097K";

static const int ALGORITHM_ITERATIONS = 100;
static const int DIMENSION = 3;
static const int NNB = 4;

static const float GAMMA = 1.4f;
static const float FAR_FIELD_MACH = 1.2f;
static const int RK = 3;
static const int VAR_DENSITY = 0;
static const int VAR_MOMENTUM = 1;
static const int VAR_DENSITY_ENERGY = (VAR_MOMENTUM + DIMENSION);
static const int NVAR = (VAR_DENSITY_ENERGY + 1);


struct Point {
    float x;
    float y;
    float z;
};

Cfd::Cfd(std::shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

Cfd::~Cfd() {

}

bool Cfd::LoadInputData() {
    ifstream fileStream(CL_DATA_PATH_PREFIX + CFD_DATASET, ios::in);
    string inputData = "";

    fileStream.seekg(0, ios::end);
    inputData.resize(static_cast<size_t>(fileStream.tellg()));
    fileStream.seekg(0, ios::beg);
    fileStream.read(&inputData[0], inputData.size());
    fileStream.close();

    stringstream stream(inputData, ios::in);

    stream >> _pointCount;

    if (!stream.good())
        return false;

    // enough points to fill out all blocks we have to process
    _pointCountPadded = _pointCount + (_requestedWorkGroupSize - (_pointCount % _requestedWorkGroupSize)) % _requestedWorkGroupSize;

    // memory allocation
    _areas = new float[_pointCountPadded];
    _surroundingElementsCounters = new int[_pointCountPadded * NNB];
    _normalVectors = new float[_pointCountPadded * DIMENSION * NNB];

    bool failed = false;
    for (int i = 0; i < _pointCount && !failed; ++i) {
        stream >> _areas[i];

        for (int j = 0; j < NNB; ++j) {
            stream >> _surroundingElementsCounters[j * _pointCountPadded + i];

            // convert number according according to original code
            if (_surroundingElementsCounters[j * _pointCountPadded + i] < 0) {
                _surroundingElementsCounters[j * _pointCountPadded + i] = -2;
            } else {
                _surroundingElementsCounters[j * _pointCountPadded + i]--;
            }

            for (int k = 0; k < DIMENSION; ++k) {
                int index = (k * NNB + j) * _pointCountPadded + i;
                stream >> _normalVectors[index];
                _normalVectors[index] = -_normalVectors[index];
            }

            if (!stream.good()) {
                failed = true;
                break;
            }
        }
    }

    if (failed) {
        delete[] _areas;
        delete[] _surroundingElementsCounters;
        delete[] _normalVectors;

        _areas = nullptr;
        _surroundingElementsCounters = nullptr;
        _normalVectors = nullptr;
        return false;
    }

    // fill in padding
    for (int i = _pointCount; i < _pointCountPadded; ++i) {
        _areas[i] = _areas[_pointCount - 1];
        for (int j = 0; j < NNB; ++j) {
            _surroundingElementsCounters[j * _pointCountPadded + i] = _surroundingElementsCounters[j * _pointCountPadded + _pointCount - 1];
            for (int k = 0; k < DIMENSION; ++k) {
                _normalVectors[(k * NNB + j) * _pointCountPadded + i] = _normalVectors[(k * NNB + j) * _pointCountPadded + _pointCount - 1];
            }
        }
    }

    return true;
}

void Cfd::InitFarFieldData() {
    float ffVariable[NVAR];

    ffVariable[VAR_DENSITY] = 1.4f;
    float ffPressure = 1.0f;
    float ffSpeedOfSound = sqrt(GAMMA * ffPressure / ffVariable[VAR_DENSITY]);
    float ffSpeed = FAR_FIELD_MACH * ffSpeedOfSound;

    Point ffVelocity;
    ffVelocity.x = ffSpeed;
    ffVelocity.y = 0.0f;
    ffVelocity.z = 0.0f;

    ffVariable[VAR_MOMENTUM] = ffVariable[VAR_DENSITY] * ffSpeed;
    ffVariable[VAR_MOMENTUM + 1] = ffVariable[VAR_MOMENTUM + 2] = 0.0f;

    Point ffMomentum;
    ffMomentum.x = ffVariable[VAR_MOMENTUM];
    ffMomentum.y = 0.0f;
    ffMomentum.z = 0.0f;

    Point ffFluxContX, ffFluxContY, ffFluxContZ, ffFluxContEnergy;

    ffFluxContX.x = ffVelocity.x * ffMomentum.x + ffPressure;
    ffFluxContX.y = 0.0f;
    ffFluxContX.z = 0.0f;

    ffFluxContY.x = ffFluxContX.y;
    ffFluxContY.y = ffPressure;
    ffFluxContY.z = 0.0f;

    ffFluxContZ.x = 0.0f;
    ffFluxContZ.y = 0.0f;
    ffFluxContZ.z = ffPressure;

    float dep = ffVariable[VAR_DENSITY_ENERGY] + ffPressure;
    ffFluxContEnergy.x = ffVelocity.x * dep;
    ffFluxContEnergy.y = 0.0f;
    ffFluxContEnergy.z = 0.0f;


    _ff_variableBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, NVAR * sizeof(float));
    _ff_fluxXBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(Point));
    _ff_fluxYBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(Point));
    _ff_fluxZBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(Point));
    _ff_fluxEnergyBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(Point));

    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_ff_variableBuffer, CL_TRUE, 0, NVAR * sizeof(float), static_cast<void*>(ffVariable));
    queue.enqueueWriteBuffer(*_ff_fluxXBuffer, CL_TRUE, 0, sizeof(Point), static_cast<void*>(&ffFluxContX));
    queue.enqueueWriteBuffer(*_ff_fluxYBuffer, CL_TRUE, 0, sizeof(Point), static_cast<void*>(&ffFluxContY));
    queue.enqueueWriteBuffer(*_ff_fluxZBuffer, CL_TRUE, 0, sizeof(Point), static_cast<void*>(&ffFluxContZ));
    queue.enqueueWriteBuffer(*_ff_fluxEnergyBuffer, CL_TRUE, 0, sizeof(Point), static_cast<void*>(&ffFluxContEnergy));

    queue.finish();
}

int Cfd::InitKernelsAndBuffers() {
    string compilerParams = GetCompilerFlags<float>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "cfd.cl", compilerParams);

    cl_int status = CL_SUCCESS;
    _memsetKernel = make_shared<cl::Kernel>(*_program, "memset_kernel", &status);
    CHECK_RETURN_ERROR(status);
    _initializeVariablesKernel = make_shared<cl::Kernel>(*_program, "initialize_variables", &status);
    CHECK_RETURN_ERROR(status);
    _computeStepFactorKernel = make_shared<cl::Kernel>(*_program, "compute_step_factor", &status);
    CHECK_RETURN_ERROR(status);
    _computeFluxKernel = make_shared<cl::Kernel>(*_program, "compute_flux", &status);
    CHECK_RETURN_ERROR(status);
    _timeStepKernel = make_shared<cl::Kernel>(*_program, "time_step", &status);
    CHECK_RETURN_ERROR(status);

    _areasBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(float) * _pointCountPadded);
    _surroundingElementsCountersBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(int) * _pointCountPadded * NNB);
    _normalVectorsBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, sizeof(float) * _pointCountPadded * DIMENSION * NNB);

    _variablesBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _pointCountPadded * NVAR * sizeof(float));
    _oldVariablesBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _pointCountPadded * NVAR * sizeof(float));
    _fluxesBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _pointCountPadded * NVAR * sizeof(float));
    _stepFactorsBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _pointCountPadded * sizeof(float));

    return 0;
}

void Cfd::InitDeviceMemory() {
    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_areasBuffer, CL_TRUE, 0, sizeof(float) * _pointCountPadded, static_cast<void*>(_areas));
    queue.enqueueWriteBuffer(*_surroundingElementsCountersBuffer, CL_TRUE, 0, sizeof(int) * _pointCountPadded * NNB, static_cast<void*>(_surroundingElementsCounters));
    queue.enqueueWriteBuffer(*_normalVectorsBuffer, CL_TRUE, 0, sizeof(float) * _pointCountPadded * NNB * DIMENSION, static_cast<void*>(_normalVectors));
    queue.finish();

    cl_int status = CL_SUCCESS;

    cl::NDRange local(_requestedWorkGroupSize);
    cl::NDRange global(_pointCountPadded);
    cl::Event event;

    _initializeVariablesKernel->setArg(0, *_variablesBuffer);
    _initializeVariablesKernel->setArg(1, *_ff_variableBuffer);
    _initializeVariablesKernel->setArg(2, _pointCountPadded);

    status = queue.enqueueNDRangeKernel(*_initializeVariablesKernel, cl::NullRange, global, local, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    _initializeVariablesKernel->setArg(0, *_oldVariablesBuffer);
    status = queue.enqueueNDRangeKernel(*_initializeVariablesKernel, cl::NullRange, global, local, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    _initializeVariablesKernel->setArg(0, *_fluxesBuffer);
    status = queue.enqueueNDRangeKernel(*_initializeVariablesKernel, cl::NullRange, global, local, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    cl_short value = 0;
    cl_int sizeOfBuffer = sizeof(float) * _pointCountPadded;
    _memsetKernel->setArg(0, *_stepFactorsBuffer);
    _memsetKernel->setArg(1, value);
    _memsetKernel->setArg(2, sizeOfBuffer);

    status = queue.enqueueNDRangeKernel(*_memsetKernel, cl::NullRange, global, local, nullptr, &event);
    WAIT_AND_CHECK(event, status);

    queue.finish();
}

void Cfd::SetKernelArguments() {
    _computeStepFactorKernel->setArg(0, *_variablesBuffer);
    _computeStepFactorKernel->setArg(1, *_areasBuffer);
    _computeStepFactorKernel->setArg(2, *_stepFactorsBuffer);
    _computeStepFactorKernel->setArg(3, _pointCountPadded);

    _computeFluxKernel->setArg(0, *_surroundingElementsCountersBuffer);
    _computeFluxKernel->setArg(1, *_normalVectorsBuffer);
    _computeFluxKernel->setArg(2, *_variablesBuffer);
    _computeFluxKernel->setArg(3, *_ff_variableBuffer);
    _computeFluxKernel->setArg(4, *_fluxesBuffer);
    _computeFluxKernel->setArg(5, *_ff_fluxEnergyBuffer);
    _computeFluxKernel->setArg(6, *_ff_fluxXBuffer);
    _computeFluxKernel->setArg(7, *_ff_fluxYBuffer);
    _computeFluxKernel->setArg(8, *_ff_fluxZBuffer);
    _computeFluxKernel->setArg(9, _pointCountPadded);

    _timeStepKernel->setArg(1, _pointCountPadded);
    _timeStepKernel->setArg(2, *_oldVariablesBuffer);
    _timeStepKernel->setArg(3, *_variablesBuffer);
    _timeStepKernel->setArg(4, *_stepFactorsBuffer);
    _timeStepKernel->setArg(5, *_fluxesBuffer);
}

void Cfd::RunInternal() {
    cl::CommandQueue& queue = _controller->Queue();
    cl::Event event;
    cl::NDRange local(_requestedWorkGroupSize);
    cl::NDRange global(_pointCountPadded);
    cl_int status = CL_SUCCESS;
    cl_long startTime, endTime;

    int64_t timeComputeStepFactorCPU = 0;
    int64_t timeComputeStepFactorGPU = 0;
    int64_t timeComputeFluxCPU = 0;
    int64_t timeComputeFluxGPU = 0;
    int64_t timeTimeStepCPU = 0;
    int64_t timeTimeStepGPU = 0;

    for (int i = 0; i < ALGORITHM_ITERATIONS; ++i) {
        // backup variables
        queue.enqueueCopyBuffer(*_variablesBuffer, *_oldVariablesBuffer, 0, 0, _pointCountPadded * NVAR * sizeof(float), nullptr, &event);
        event.wait();
        queue.finish();

        // compute step factors
        _timer.Remember();
        status = queue.enqueueNDRangeKernel(*_computeStepFactorKernel, cl::NullRange, global, local, nullptr, &event);
        WAIT_AND_CHECK(event, status);
        timeComputeStepFactorCPU += _timer.Diff();

        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
        timeComputeStepFactorGPU += (endTime - startTime);

        for (int j = 0; j < RK; ++j) {
            _timer.Remember();
            status = queue.enqueueNDRangeKernel(*_computeFluxKernel, cl::NullRange, global, local, nullptr, &event);
            WAIT_AND_CHECK(event, status);
            timeComputeFluxCPU += _timer.Diff();

            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
            timeComputeFluxGPU += (endTime - startTime);

            _timeStepKernel->setArg(0, j);
            _timer.Remember();
            status = queue.enqueueNDRangeKernel(*_timeStepKernel, cl::NullRange, global, local, nullptr, &event);
            WAIT_AND_CHECK(event, status);
            timeTimeStepCPU += _timer.Diff();

            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
            timeTimeStepGPU += (endTime - startTime);
        }
    }

    cout << "ComputeStepFactor, CPU: " << timeComputeStepFactorCPU << ", GPU: " << timeComputeStepFactorGPU << endl;
    cout << "ComputeFlux,       CPU: " << timeComputeFluxCPU << ", GPU: " << timeComputeFluxGPU << endl;
    cout << "TimeStep,          CPU: " << timeTimeStepCPU << ", GPU: " << timeTimeStepGPU << endl;
    cout << "Total,             CPU: " << (timeTimeStepCPU + timeComputeFluxCPU + timeComputeStepFactorCPU)
        << ", GPU: " << (timeTimeStepGPU + timeComputeFluxGPU + timeComputeStepFactorGPU) << endl;

}

void Cfd::Cleanup() {
    if (_areas != nullptr)
        delete[] _areas;
    if (_surroundingElementsCounters != nullptr)
        delete[] _surroundingElementsCounters;
    if (_surroundingElementsCounters != nullptr)
        delete[] _normalVectors;

    _areas = nullptr;
    _surroundingElementsCounters = nullptr;
    _normalVectors = nullptr;

    _ff_variableBuffer.reset();
    _ff_fluxXBuffer.reset();
    _ff_fluxYBuffer.reset();
    _ff_fluxZBuffer.reset();
    _ff_fluxEnergyBuffer.reset();
    _areasBuffer.reset();
    _surroundingElementsCountersBuffer.reset();
    _normalVectorsBuffer.reset();
    _variablesBuffer.reset();
    _oldVariablesBuffer.reset();
    _fluxesBuffer.reset();
    _stepFactorsBuffer.reset();

    _memsetKernel.reset();
    _initializeVariablesKernel.reset();
    _computeStepFactorKernel.reset();
    _computeFluxKernel.reset();
    _timeStepKernel.reset();

    _program.reset();
}


void Cfd::Run() {
    cout << "Computational Fluid Dynamics Test:" << endl;

    int workGroupSizes[] = { 32, 64, 128, 192, 256, 512 };

    for (auto& workGroupSize : workGroupSizes) {
        RequestWorkGroupSize(workGroupSize);

        cout << "WorkGroupSize: " << workGroupSize << endl;

        InitFarFieldData();
        if (LoadInputData() && InitKernelsAndBuffers() == 0) {
            InitDeviceMemory();
            SetKernelArguments();
            RunInternal();
        }
        Cleanup();
    }

    cout << endl;
}