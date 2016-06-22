#ifndef __BENCH_COMPUTECONTROLLER_HPP
#define __BENCH_COMPUTECONTROLLER_HPP

#include <memory>
#include <string>
#include <vector>

#include "clglobal.hpp"

/**
 * This class handles opencl resource management.
 * It provides functions to select platforms and devices.
 * Additionally it also allows to manage compiled binaries and 
 * querying hardware features.
 */
class ComputeController {
private:
    cl::Platform _selectedPlatform;
    cl::Device _selectedDevice;
    std::vector<cl::Device> _devices;
    
    cl::Context _context;
    cl::CommandQueue _queue;

    std::shared_ptr<cl::Program> BuildCommon(std::shared_ptr<cl::Program> program, const std::string& compilerParams);

    bool _saveProgramBinaries;

public:
    explicit ComputeController();
    virtual ~ComputeController();

    /**
     * Tell the compute controller if compiled kernels should be dumped into a file.
     */
    void SetSaveProgramBinaries(bool saveProgramBinaries);

    /**
     * Prints information about platforms on the terminal and asks the user to select one.
     * Does the same for devices after that.
     * 
     * @param showDetails tells the function to print additional information like vendor, version, profile and extensions
     * @return zero on success
     */
    int SelectDeviceDialog(bool showDetails);

    /**
     * Selected platform.
     * 
     * @return reference to the c++ wrapper instance around cl_platform_id
     */
    cl::Platform& SelectedPlatform();

    /**
     * Selected Device.
     *
     * @return reference to the c++ wrapper instance around cl_device_id
     */
    cl::Device& SelectedDevice();

    /**
     * Context.
     *
     * @return reference to the c++ wrapper instance around cl_context
     */
    cl::Context& Context();

    /**
     * Queue waiting for commands for the selected platform and device.
     *
     * @return reference to the c++ wrapper instance around cl_command_queue
     */
    cl::CommandQueue& Queue();

    /**
     * Checks whether the device supports double precision floating point procesing.
     *
     * @return true if device supports cl_khr_fp64 or cl_amd_fp64
     */
    bool SupportsDoublePrecision();

    /**
     * Checks whether the device supports the extension given by the name.
     *
     * @param name identifier of an extensino (e.g. cl_khr_spir)
     * @return true if extension is supported
     */
    bool HasExtension(std::string name);

    /**
     * Build kernel given by a file path.
     *
     * @param path path a file containing opencl c code.
     * @param compilerParams compiler flags
     * @return the compiled program instance (wrapper around cl_program)
     */
    std::shared_ptr<cl::Program> BuildFromSource(const std::string& path, const std::string& compilerParams = "");

    /**
     * Build kernel from the source given by a string.
     * 
     * @param code the given kernel code
     * @param compilerParams compiler flagsg
     * @return the compiled program instance (wrapper around cl_program)
     */
    std::shared_ptr<cl::Program> BuildFromSourceStr(const std::string& code, const std::string& compilerParams = "");

    /**
     * Build kernel from a previously dumped binary.
     *
     * @param path path to a compiled kernel image.
     * @param compilerParams compiler flags
     * @return the compiled program instance (wrapper around cl_program)
     */
    std::shared_ptr<cl::Program> BuildFromBinary(const std::string& path, const std::string& compilerParams = "");

    /**
     * Build kernel from a binary stored in memory.
     *
     * @param binary to a compiled kernel image.
     * @param compilerParams compiler flags
     * @return the compiled program instance (wrapper around cl_program)
     */
    std::shared_ptr<cl::Program> BuildFromBinary(std::vector<char>& binary, const std::string& compilerParams = "");

    /**
     * Save a compiled kernel image to the hard disk.
     *
     * @param program instance
     * @param name identifier of the kernel file
     * @param path where to store the kernel image
     * @param compilerParams compiler flags (used to add postfixes to the filenames like the used data type)
     */
    void SavePlatformSpecificBinary(std::shared_ptr<cl::Program> program, const std::string& name, const std::string& path, const std::string& compilerParams);

    /**
     * Save compiled kernel image to a memory buffer.
     *
     * @param binbary to a compiled kernel image.
     * @param compilerParams compiler flags
     */
    void SavePlatformSpecificBinary(std::shared_ptr<cl::Program> program, std::vector<char>& data);
};

#endif // __BENCH_COMPUTECONTROLLER_HPP
