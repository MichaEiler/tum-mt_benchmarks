#ifdef VTYPE_FLOAT
#define VTYPE float
#elif VTYPE_DOUBLE_KHR
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VTYPE double
#elif VTYPE_DOUBLE_AMD
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VTYPE double
#elif VTYPE_INT
#define VTYPE int
#elif VTYPE_LONG
#define VTYPE long
#endif

__kernel
void vecadd(__global VTYPE *A, __global VTYPE *B, __global VTYPE *C) {
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

__kernel
void vecmul(__global VTYPE *A, __global VTYPE *B, __global VTYPE *C) {
    int i = get_global_id(0);
    C[i] = A[i] * B[i];
}

__kernel
void vecdiv(__global VTYPE *A, __global VTYPE *B, __global VTYPE *C) {
    int i = get_global_id(0);
    C[i] = A[i] / B[i];
}

