#include "spmv.hpp"

#include <algorithm>
#include <iostream>
#include <random>

#include "../clglobal.hpp"
#include "../computecontroller.hpp"

using namespace benchmarks;
using namespace std;

static const int MAXVAL = 10;
static const int RANDOM_SEED = 85733;
static const int SPMV_DIMENSION = 4096; //1024, 8192, 12288, 16384
static const int TEST_ITERATIONS = 100;

Spmv::Spmv(std::shared_ptr<ComputeController> controller)
    : BenchmarkBase(controller) {

}

Spmv::~Spmv() {

}

template <typename TItem>
void Spmv::InitMatrixRowMajor(std::vector<TItem>& values, std::vector<int>& columnIds, 
                            std::vector<int>& rowDelimiters, std::vector<int>& rowLengths) {
    default_random_engine randomEngine(RANDOM_SEED);
    uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
    uniform_real_distribution<TItem> valueDistribution(0.0, 1.0);

    int numberOfRows = SPMV_DIMENSION + (_requestedWorkGroupSize - SPMV_DIMENSION % _requestedWorkGroupSize) % _requestedWorkGroupSize;
    int numberOfItems = (numberOfRows * numberOfRows) / 10;//100;

    double probability = static_cast<double>(numberOfItems) / static_cast<double>(numberOfRows * numberOfRows);

    values.resize(numberOfItems);
    columnIds.resize(numberOfItems);
    rowDelimiters.resize(numberOfRows + 1);
    rowLengths.resize(numberOfRows);

    // fill in values
    for (int i = 0; i < numberOfItems; ++i)
        values[i] = valueDistribution(randomEngine) * MAXVAL;

    // assign values to cells
    int valueIterator = 0;
    bool fillRemaining = false;
    for (int i = 0; i < numberOfRows; ++i) {
        rowDelimiters[i] = valueIterator;
        for (int j = 0; j < numberOfRows; ++j) {
            int numEntriesLeft = (numberOfRows * numberOfRows) - ((i * numberOfRows) + j);
            int needToAssigned = numberOfItems - valueIterator;
            if (numEntriesLeft <= needToAssigned)
                fillRemaining = true;

            if ((valueIterator < numberOfItems && probabilityDistribution(randomEngine) < probability) || fillRemaining) {
                columnIds[valueIterator] = j;
                valueIterator++;
            }
        }
    }
    // number of non-zeros at the end of rowDelimiters
    rowDelimiters[numberOfRows] = numberOfItems;

    // assign rowLengths information
    for (int i = 0; i < numberOfRows; ++i)
        rowLengths[i] = rowDelimiters[i + 1] - rowDelimiters[i];
}

template <typename TItem>
void Spmv::ConvertToPaddedRowMajor(std::vector<TItem>& values, std::vector<int>& columnIds, 
                        std::vector<int>& rowDelimiters, std::vector<int>& rowLengths,
                        std::vector<TItem>& valuesPadded, std::vector<int>& columnIdsPadded, int maxRowLength) {
    int numberOfRows = static_cast<int>(rowDelimiters.size()) - 1;

    columnIdsPadded.resize(numberOfRows * maxRowLength);
    valuesPadded.resize(numberOfRows * maxRowLength);

    // overwrite everything with zeros as default value
    for (int i = 0; i < numberOfRows * maxRowLength; ++i) {
        columnIdsPadded[i] = 0;
        valuesPadded[i] = static_cast<TItem>(0.0f);
    }

    // insert all values from unpadded vectors
    int valueIterator = 0;
    for (int i = 0; i < numberOfRows; ++i) {
        for (int j = 0; j < rowLengths[i]; ++j) {
            columnIdsPadded[i * maxRowLength + j] = columnIds[valueIterator];
            valuesPadded[i * maxRowLength + j] = values[valueIterator]; 
            valueIterator++;
        }
    }
}

template <typename TItem>
void Spmv::ConvertToPaddedColumnMajor(std::vector<TItem>& valuesPadded, std::vector<int>& columnIdsPadded,
                        std::vector<TItem>& valuesColumnMajorPadded, std::vector<int>& columnIdsColumnMajorPadded,
                        int maxRowLength) {
    valuesColumnMajorPadded.resize(valuesPadded.size());
    columnIdsColumnMajorPadded.resize(valuesPadded.size());

    int numberOfRows = static_cast<int>(valuesPadded.size()) / maxRowLength;

    for (int i = 0; i < numberOfRows; ++i) {
        for (int j = 0; j < maxRowLength; ++j) {
            int srcIndex = i * maxRowLength + j;
            int dstIndex = j * numberOfRows + i;
            valuesColumnMajorPadded[dstIndex] = valuesPadded[srcIndex];
            columnIdsColumnMajorPadded[dstIndex] = columnIdsPadded[srcIndex];
        }
    }

}

template <typename TItem>
void Spmv::InitVector(TItem *values, int n) {
    static default_random_engine randomEngine(~RANDOM_SEED);
    static uniform_real_distribution<TItem> valueDistribution(0.0, 1.0);

    // store random values
    for (int i = 0; i < n; ++i) {
        values[i] = valueDistribution(randomEngine) * MAXVAL;
    }
}

template <typename TItem>
int Spmv::InitContext() {
    string compilerParams = GetCompilerFlags<TItem>();
    _program = _controller->BuildFromSource(CL_SRC_PATH_PREFIX + "spmv.cl", compilerParams);

    if (_program.get() == nullptr)
        return -1;

    cl_int status = CL_SUCCESS;
    _ellpackRowKernel = make_shared<cl::Kernel>(*_program, "spmv_ellpackr_kernel_rowmajor", &status);
    CHECK_RETURN_ERROR(status);
    _ellpackKernel = make_shared<cl::Kernel>(*_program, "spmv_ellpackr_kernel", &status);
    CHECK_RETURN_ERROR(status);

    // init input matrix
    std::vector<TItem> valuesRMU;
    std::vector<int> columnIdsRMU, rowDelimiters, rowLengths;
    InitMatrixRowMajor<TItem>(valuesRMU, columnIdsRMU, rowDelimiters, rowLengths);
    // convert to padded row major
    _maxRowLength = *max_element(rowLengths.begin(), rowLengths.end());
    std::vector<TItem> valuesRMP;

    std::vector<int> columnIdsRMP;
    ConvertToPaddedRowMajor(valuesRMU, columnIdsRMU, rowDelimiters, rowLengths, valuesRMP, columnIdsRMP, _maxRowLength);

    // convert to padded column major
    std::vector<TItem> valuesCMP;
    std::vector<int> columnIdsCMP;

    ConvertToPaddedColumnMajor(valuesRMP, columnIdsRMP, valuesCMP, columnIdsCMP, _maxRowLength);

    // init input vector
    _numberOfRows = static_cast<int>(rowDelimiters.size()) - 1;
    std::vector<TItem> inputVector;
    inputVector.resize(_numberOfRows);
    InitVector(&inputVector[0], _numberOfRows);

    // create buffers
    _inputValueBufferCMP = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _maxRowLength * _numberOfRows * sizeof(TItem));
    _inputValueBufferRMP = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _maxRowLength * _numberOfRows * sizeof(TItem));
    _inputColumnsBufferCMP = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _maxRowLength * _numberOfRows * sizeof(int));
    _inputColumnsBufferRMP = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _maxRowLength * _numberOfRows * sizeof(int));
    _inputRowLengthsBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _numberOfRows * sizeof(int));
    _inputVectorBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _numberOfRows * sizeof(TItem));
    _outputVectorBuffer = make_shared<cl::Buffer>(_controller->Context(), CL_MEM_READ_WRITE, _numberOfRows * sizeof(TItem));

    // copy data to device
    cl::CommandQueue& queue = _controller->Queue();
    queue.enqueueWriteBuffer(*_inputValueBufferCMP, CL_TRUE, 0, _maxRowLength * _numberOfRows * sizeof(TItem), &valuesCMP[0]);
    queue.enqueueWriteBuffer(*_inputValueBufferRMP, CL_TRUE, 0, _maxRowLength * _numberOfRows * sizeof(TItem), &valuesRMP[0]);
    queue.enqueueWriteBuffer(*_inputColumnsBufferCMP, CL_TRUE, 0, _maxRowLength * _numberOfRows * sizeof(int), &columnIdsCMP[0]);
    queue.enqueueWriteBuffer(*_inputColumnsBufferRMP, CL_TRUE, 0, _maxRowLength * _numberOfRows * sizeof(int), &columnIdsRMP[0]);
    queue.enqueueWriteBuffer(*_inputRowLengthsBuffer, CL_TRUE, 0, _numberOfRows * sizeof(int), &rowLengths[0]);
    queue.enqueueWriteBuffer(*_inputVectorBuffer, CL_TRUE, 0, _numberOfRows * sizeof(TItem), &inputVector[0]);
    queue.finish();

    return 0;
}

void Spmv::SetKernelArguments() {
    _ellpackKernel->setArg(0, *_inputValueBufferCMP);
    _ellpackKernel->setArg(1, *_inputVectorBuffer);
    _ellpackKernel->setArg(2, *_inputColumnsBufferCMP);
    _ellpackKernel->setArg(3, *_inputRowLengthsBuffer);
    _ellpackKernel->setArg(4, _numberOfRows);
    _ellpackKernel->setArg(5, *_outputVectorBuffer);

    _ellpackRowKernel->setArg(0, *_inputValueBufferRMP);
    _ellpackRowKernel->setArg(1, *_inputVectorBuffer);
    _ellpackRowKernel->setArg(2, *_inputColumnsBufferRMP);
    _ellpackRowKernel->setArg(3, *_inputRowLengthsBuffer);
    _ellpackRowKernel->setArg(4, _numberOfRows);
    _ellpackRowKernel->setArg(5, _maxRowLength);
    _ellpackRowKernel->setArg(6, *_outputVectorBuffer);  
}

template <typename TItem>
void Spmv::RunInternal() {
    std::vector<TItem> resultCM;
    std::vector<TItem> resultRM;

    resultCM.resize(_numberOfRows);
    resultRM.resize(_numberOfRows);

    cl::CommandQueue& queue = _controller->Queue();
    cl::NDRange local(_requestedWorkGroupSize);
    cl::NDRange global(_numberOfRows);
    cl_int status = CL_SUCCESS;

    string testName = "Column Major, ";
    PerformTest([&](cl::Event& event) -> void {
            status = queue.enqueueNDRangeKernel(*_ellpackKernel, cl::NullRange, global, local, nullptr, &event);
            CHECK(status);
        }, testName, TEST_ITERATIONS);

    // copy result back
    queue.enqueueReadBuffer(*_outputVectorBuffer, CL_TRUE, 0, _numberOfRows * sizeof(TItem), &resultCM[0]);
    queue.finish();

    testName = "Row Major,    ";
    PerformTest([&](cl::Event& event) -> void {
            status = queue.enqueueNDRangeKernel(*_ellpackRowKernel, cl::NullRange, global, local, nullptr, &event);
            CHECK(status);
        }, testName, TEST_ITERATIONS);

    // copy result back
    queue.enqueueReadBuffer(*_outputVectorBuffer, CL_TRUE, 0, _numberOfRows * sizeof(TItem), &resultRM[0]);
    queue.finish();

    for (int i = 0; i < _numberOfRows; ++i) {
        if (resultCM[i] != resultRM[i]) {
            cout << resultCM[i] << " " << resultRM[i] << endl;
            cerr << "Results are not equal." << endl;
            return;
        }
    }
}

void Spmv::Cleanup() {
    _inputValueBufferCMP.reset();
    _inputValueBufferRMP.reset();
    _inputColumnsBufferCMP.reset();
    _inputColumnsBufferRMP.reset();
    _inputRowLengthsBuffer.reset();
    _inputVectorBuffer.reset();
    _outputVectorBuffer.reset();

    _ellpackKernel.reset(),
    _ellpackRowKernel.reset();
    _program.reset();
}

void Spmv::Run() {
    cout << "Sparse Matrix Vector Multiplication:" << endl;

    int workGroupSizes[] = { 32, 64, 128, 192, 256};//, 512, 1024 };

    for (auto& workGroupSize : workGroupSizes) {
        RequestWorkGroupSize(workGroupSize);
        cout << "Spmv::Ellpack<float>, WorkGroupSize: " << workGroupSize << endl;
        if (InitContext<float>() == 0) {
            SetKernelArguments();
            RunInternal<float>();
            Cleanup();
        }

        if (_controller->SupportsDoublePrecision()) {
            cout << "Spmv::Ellpack<double>, WorkGroupSize: " << workGroupSize << endl;
            if (InitContext<double>() == 0) {
                SetKernelArguments();
                RunInternal<double>();
                Cleanup();
            }
        }
    }

    cout << endl;
}
