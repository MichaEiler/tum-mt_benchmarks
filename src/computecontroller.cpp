#include "computecontroller.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

ComputeController::ComputeController()
    : _selectedPlatform()
    , _selectedDevice()
    , _devices()
    , _context()
    , _queue()
    , _saveProgramBinaries(false) {

}

ComputeController::~ComputeController() {

}

void ComputeController::SetSaveProgramBinaries(bool saveProgramBinaries) {
    _saveProgramBinaries = saveProgramBinaries;
}

cl::Platform& ComputeController::SelectedPlatform() {
    return _selectedPlatform;
}

cl::Device& ComputeController::SelectedDevice() {
    return _selectedDevice;
}

cl::Context& ComputeController::Context() {
    return _context;
}

cl::CommandQueue& ComputeController::Queue() {
    return _queue;
}

int ComputeController::SelectDeviceDialog(bool showDetails) {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cout << "The following CL capable platforms are available:" << endl;
    for(size_t i = 0; i < platforms.size(); ++i) {
        string name, vendor, profile, version, extensions;
        platforms[i].getInfo(CL_PLATFORM_NAME, &name);
        platforms[i].getInfo(CL_PLATFORM_VENDOR, &vendor);
        platforms[i].getInfo(CL_PLATFORM_VERSION, &version);
        platforms[i].getInfo(CL_PLATFORM_PROFILE, &profile);
        platforms[i].getInfo(CL_PLATFORM_EXTENSIONS, &extensions);

        cout << "    [" << i << "] " << name << endl;
        if (showDetails) {
            cout << "        Vendor: " << vendor << endl;
            cout << "        Version: " << version << endl;
            cout << "        Profile: " << profile << endl;
            cout << "        Extensions: " << extensions << endl;
        }
        cout << endl;
    }

    int platformId = -1;
    while (true) {
        cout << "Choose a platform: ";
        cin >> platformId;

        if (platformId >= 0 && platformId < static_cast<int>(platforms.size()))
            break;
    }

    _selectedPlatform = platforms[platformId];

    _selectedPlatform.getDevices(CL_DEVICE_TYPE_ALL, &_devices);

    cout << endl << "The selected platform provides the following supported devices:" << endl;
    for (size_t i = 0; i < _devices.size(); ++i) {
        string name, profile, extensions, version; // add infos like work group sizes...
        _devices[i].getInfo(CL_DEVICE_NAME, &name);
        _devices[i].getInfo(CL_DEVICE_PROFILE, &profile);
        _devices[i].getInfo(CL_DEVICE_EXTENSIONS, &extensions);
        _devices[i].getInfo(CL_DEVICE_VERSION, &version);
        
        cout << "    [" << i << "] " << name << endl;
        if (showDetails) {
            cout << "        Profile: " << profile << endl;
            cout << "        Extensions: " << extensions << endl;
            cout << "        Version: " << version << endl;
        }
        cout << endl;
    }

    int deviceId = -1;
    while (true) {
        cout << "Choose a device: ";
        cin >> deviceId;
        
        if (deviceId >= 0 && deviceId < static_cast<int>(_devices.size()))
            break;
    }

    _selectedDevice = _devices[deviceId];
	cl_int status;
	_context = cl::Context(_devices, nullptr, nullptr, nullptr, &status);

	if (status != CL_SUCCESS) {
		cerr << "Initializing device failed, error code: " << status << endl;
        return -1;
	}

    _queue = cl::CommandQueue(_context, _selectedDevice, CL_QUEUE_PROFILING_ENABLE);
    return 0;
}

bool ComputeController::HasExtension(std::string name) {
    string extensions;
    _selectedDevice.getInfo(CL_DEVICE_EXTENSIONS, &extensions);
    return extensions.find(name) != string::npos;
}

bool ComputeController::SupportsDoublePrecision() {
    return HasExtension("cl_khr_fp64") || HasExtension("cl_amd_fp64");
}

shared_ptr<cl::Program> ComputeController::BuildCommon(shared_ptr<cl::Program> program, const string& compilerParams) {
    cl_int buildResult = CL_SUCCESS;
    if ((buildResult = program->build(_devices, compilerParams.c_str())) != CL_SUCCESS) {
        string value;
        cerr << "BUILD Error-Code: " << buildResult << endl;
        program->getBuildInfo<string>(_selectedDevice, CL_PROGRAM_BUILD_OPTIONS, &value);
        cerr << "BUILD_OPTIONS: " << value << endl << endl;
        program->getBuildInfo<string>(_selectedDevice, CL_PROGRAM_BUILD_LOG, &value);
        cerr << "BUILD_LOG: " << value << endl << endl;

        program.reset();
    }
    return program;
}

shared_ptr<cl::Program> ComputeController::BuildFromSourceStr(const string& code, const string& compilerParams) {
    shared_ptr<cl::Program> program = nullptr;

    cl_int status = CL_SUCCESS;
    program = shared_ptr<cl::Program>(new cl::Program(_context, code, false, &status));

    if (status == CL_SUCCESS) {
        program = BuildCommon(program, compilerParams);
    } else {
        cerr << "Could not load program" << endl;
        program.reset();
    }

    return program;
}

shared_ptr<cl::Program> ComputeController::BuildFromSource(const string& path, const string& compilerParams) {
    shared_ptr<cl::Program> program = nullptr;
    string code = "";

    ifstream stream(path, ios::in);
    if (stream) {
        stream.seekg(0, ios::end);
        code.resize(static_cast<size_t>(stream.tellg()));
        stream.seekg(0, ios::beg);
        stream.read(&code[0], code.size());
        stream.close();

        program = BuildFromSourceStr(code, compilerParams);
    } else {
        cerr << "*.cl code file not found" << endl;
    }

    if (program.get() != nullptr && _saveProgramBinaries) {
        size_t delim = path.find_last_of("/\\");
        string name = path.substr(delim);
        SavePlatformSpecificBinary(program, name + ".bin", "./", compilerParams);
    }

    return program;
}

shared_ptr<cl::Program> ComputeController::BuildFromBinary(const string& path, const string& compilerParams) {
    shared_ptr<cl::Program> program;

    size_t binarySize;
    vector<char> binary;

    // load binary file
    ifstream stream(path, ios::in);
    if (stream) {
        stream.seekg(0, ios::end);
        binarySize = static_cast<size_t>(stream.tellg());
        binary.resize(binarySize);
        stream.seekg(0, ios::beg);
        stream.read(&binary[0], binarySize);
        stream.close();

        program = BuildFromBinary(binary, compilerParams);
    } else {
        cerr << "kernel file could not be found" << endl;
    }

    return program;
}

shared_ptr<cl::Program> ComputeController::BuildFromBinary(vector<char>& binary, const string& compilerParams) {
    shared_ptr<cl::Program> program;

    // initialize parameters
    cl::Program::Binaries binaries;
    binaries.push_back(pair<const void*, size_t>(static_cast<const void*>(&binary[0]), binary.size()));
    vector<cl::Device> devices;
    devices.push_back(_selectedDevice);

    cl_int status = CL_SUCCESS;
    program = shared_ptr<cl::Program>(new cl::Program(_context, devices, binaries, nullptr, &status));

    if (status == CL_SUCCESS) {
        program = BuildCommon(program, compilerParams);
    } else {
        cerr << "Could not load program" << endl;
        program.reset();
    }

    // cleanup
    return program;
}


void ComputeController::SavePlatformSpecificBinary(shared_ptr<cl::Program> program, vector<char>& data) {
    vector<char*> binaries;
    vector<size_t> binarySizes;

    // request the size of the binaries
    if (program->getInfo(CL_PROGRAM_BINARY_SIZES, &binarySizes) != CL_SUCCESS) {
        cerr << "Could not retrieve compute binaries." << endl;
        return;
    }

    // pre allocate buffers for binaries (this is not done by the opencl-c++ api!)
    for (size_t i = 0; i < binarySizes.size(); ++i) {
        char* buffer = new char[binarySizes[i]];
        binaries.push_back(buffer);
    }

    // request binaries
    cl_int status = program->getInfo(CL_PROGRAM_BINARIES, &binaries);
    if (status != CL_SUCCESS) {
        cerr << "Could not retrieve compute binaries. Error: " << status << endl;
    } else {
        // dump binary belonging to _selectedDevice
        for (size_t i = 0; i < binaries.size(); ++i) {
            if (_selectedDevice() != _devices[i]())
                continue;

            data.resize(binarySizes[i]);
            memcpy(&data[0], binaries[i], binarySizes[i]);
            break;
        }
    }

    // free buffers
    while (binaries.size() > 0) {
        char* buffer = binaries.back();
        binaries.pop_back();
        delete[] buffer;
    }
}

void ComputeController::SavePlatformSpecificBinary(shared_ptr<cl::Program> program, const string& name, const string& path, const string& compilerParams) {
        vector<char> binaryData;
        SavePlatformSpecificBinary(program, binaryData);

        if (binaryData.size() == 0)
            return;

        string params = compilerParams;
        transform(compilerParams.begin(), compilerParams.end(), params.begin(), ::tolower);

        string postfix = "";
        auto availableTypes = { "float", "double", "int", "long" };
        for ( const auto& type : availableTypes ) {
            if (params.find(type) != string::npos)
                postfix = string(".") + type;	
        }

        fstream outputStream(path + name + postfix, ios::binary | ios::out);
        if (outputStream) {
            outputStream.write(&binaryData[0], binaryData.size());
            outputStream.close();
        } else {
            cerr << "Failed to open output stream to store kernel binary." << endl;
        }
}


