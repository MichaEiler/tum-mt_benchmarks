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

#define BLOCK_DIMENSION 16

// based on nvidia sample code
__kernel void
transpose_optimized(__global VTYPE  *source, __global VTYPE  *destination, int height, int width){
    __local VTYPE buffer[BLOCK_DIMENSION * BLOCK_DIMENSION];

    uint columnIndex = get_global_id(0);
    uint rowIndex = get_global_id(1);

    if ((columnIndex < width) && (rowIndex < height)) {
        uint index = rowIndex * width + columnIndex;
        buffer[get_local_id(1) * BLOCK_DIMENSION + get_local_id(0)] = source[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    columnIndex = get_group_id(1) * BLOCK_DIMENSION + get_local_id(0);
    rowIndex = get_group_id(0) * BLOCK_DIMENSION + get_local_id(1);

    if ((rowIndex < width) && (columnIndex < height)) {
        uint index = rowIndex * height + columnIndex;
        destination[index] = buffer[get_local_id(0) * BLOCK_DIMENSION + get_local_id(1)];
    }
}

__kernel void
transpose_simple(__global VTYPE  *source, __global VTYPE  *destination, int height, int width) {
    __private uint sourceId = get_global_id(0);

    if (sourceId >= width * height)
        return;

    __private uint sourceRow = sourceId / width;
    __private uint sourceColumn = sourceId % width;

    destination[sourceColumn * height + sourceRow] = source[sourceId];
}
