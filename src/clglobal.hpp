#ifndef __BENCH_CLGLOBAL_HPP
#define __BENCH_CLGLOBAL_HPP

#include <string>

#define CL_HPP_MINIMUM_OPENCL_VERSION	120
#define CL_HPP_TARGET_OPENCL_VERSION	120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY	1

#ifdef USE_CL2_HEADERS
#include <CL/cl2.hpp>
#else
#include <CL/cl.hpp>
#endif

const std::string CL_SRC_PATH_PREFIX("../src/cl/");
const std::string CL_DATA_PATH_PREFIX("../data/");

#define CHECK(a) if (a != 0) {\
    std::cerr << "Error " << a << " in " << __FILE__ << " on line: " << __LINE__ << std::endl;\
    return; \
}

#define CHECK_RETURN_ERROR(a) if (a != 0) {\
    std::cerr << "Error " << a << " in " << __FILE__ << " on line: " << __LINE__ << std::endl;\
    return a; \
}

#define WAIT_AND_CHECK(event, status) if (status != CL_SUCCESS) {\
    std::cerr << "Error " << status << " in " << __FILE__ << " on line: " << __LINE__ << std::endl;\
    return;\
}\
event.wait();

#endif // __BENCH_CLGLOBAL_HPP
