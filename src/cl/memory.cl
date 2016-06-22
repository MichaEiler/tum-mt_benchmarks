#ifdef VTYPE_FLOAT
#define VTYPE float
#define VTYPE4 float4
#elif VTYPE_DOUBLE_KHR
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VTYPE double
#define VTYPE4 double4
#elif VTYPE_DOUBLE_AMD
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VTYPE double
#define VTYPE4 double4
#elif VTYPE_INT
#define VTYPE int
#define VTYPE4 int4
#elif VTYPE_LONG
#define VTYPE long
#define VTYPE4 long4
#endif

__kernel
void write_to_buffer_loop_simple(__global VTYPE *buffer, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    __private uint end = index + blockSize;

    for (; index < length && index < end; ++index) {
        buffer[index] = (VTYPE)index;
    }
}

__kernel
void write_to_buffer_loop(__global VTYPE *buffer, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    __private uint end = index + blockSize;

    for (; index < length && index < end; index += 4) {
        buffer[index] = (VTYPE)index;
        buffer[index + 1] = (VTYPE)index + 1;
        buffer[index + 2] = (VTYPE)index + 2;
        buffer[index + 3] = (VTYPE)index + 3;
    }
}

__kernel
void write_to_buffer_loop_and_vectorized(__global VTYPE *buffer, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    __private uint end = index + blockSize;

    for (; index < length && index < end; index += 4) {
    	VTYPE4 tmp = (VTYPE4)(index, index + 1, index + 2, index + 3);
    	vstore4(tmp, 0, &buffer[index]);
    }
}

__kernel
void write_to_buffer_single(__global VTYPE *buffer, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    if (index < length)
    	buffer[index] = (VTYPE)index;
}

__kernel
void write_to_buffer_vectorized(__global VTYPE *buffer, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    if (index < length) {
    	VTYPE4 tmp = (VTYPE4)(index, index + 1, index + 2, index + 3);
    	vstore4(tmp, 0, &buffer[index]);
    }
}

__kernel
void read_from_buffer_vectorized(__global VTYPE *buffer, __global VTYPE *target, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    __private uint end = index + blockSize;

    __private VTYPE4 sum = (VTYPE4)(0.0, 0.0, 0.0, 0.0);
    __private VTYPE4 tmp;

    for (; index < end && index < length; index += 4) {
        tmp = vload4(0, &buffer[index]);
        sum += tmp;
    }

    target[get_global_id(0)] = sum.x + sum.y + sum.z + sum.w;
}

__kernel
void read_from_buffer_loop_simple(__global VTYPE *buffer, __global VTYPE *target, int blockSize, int length) {
    __private uint index = get_global_id(0) * blockSize;
    __private uint end = index + blockSize;

    __private VTYPE sum = 0;

    for (; index < end && index < length; ++index) {
        sum += buffer[index];
    }

    target[get_global_id(0)] = sum;
}

__kernel
void read_from_buffer_column_major(__global VTYPE *buffer, __global VTYPE *target, int rowLength, int columnLength) {
    __private uint displacement = get_global_id(0);
    __private VTYPE sum = 0;

    for (uint i = 0; i < rowLength; ++i) {
        sum += buffer[i  * columnLength + displacement];
    }

    target[get_global_id(0)] = sum;
}

__kernel
void read_from_buffer_row_major(__global VTYPE *buffer, __global VTYPE *target, int rowLength, int columnLength) {
    __private uint displacement = get_global_id(0);
    __private VTYPE sum = 0;

    for (uint i = 0; i < rowLength; ++i) {
        sum += buffer[displacement  * rowLength + i];
    }

    target[displacement] = sum;
}