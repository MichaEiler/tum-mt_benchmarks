#ifdef VTYPE_FLOAT
#define FPTYPE float
#elif VTYPE_DOUBLE_KHR
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FPTYPE double
#elif VTYPE_DOUBLE_AMD
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define FPTYPE double
#endif

// ****************************************************************************
// Function: spmv_ellpackr_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the ELLPACK-R data storage format; based on Vazquez et al (Univ. of
//   Almeria Tech Report 2009)
//
// Arguments:
//   val: array holding the non-zero values for the matrix in column
//   vec: dense vector for multiplication
//   major format and padded with zeros up to the length of longest row
//   cols: array of column indices for each element of the sparse matrix
//   rowLengths: array storing the length of each row of the sparse matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing directly
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
//
// Modifications:
//
// ****************************************************************************
__kernel void
spmv_ellpackr_kernel(__global const FPTYPE * restrict val,
                     __global const  FPTYPE * restrict vec,
                     __global const int * restrict cols,
                     __global const int * restrict rowLengths,
                     const int numberOfRows, __global FPTYPE * restrict out)
{
    int t = get_global_id(0);

    if (t < numberOfRows)
    {
        FPTYPE result = 0.0;
        int max = rowLengths[t];
        for (int i = 0; i < max; i++)
        {
            int ind = i * numberOfRows + t;  // values stored in column major layout
            result += val[ind] * vec[cols[ind]];
        }
        out[t] = result;
    }
}

__kernel void
spmv_ellpackr_kernel_rowmajor(  __global const FPTYPE * restrict val,
                                __global const  FPTYPE * restrict vec,
                                __global const int * restrict cols,
                                __global const int * restrict rowLengths,
                                const int numberOfRows, const int maxRowLength,
                                __global FPTYPE * restrict out)
{
    int t = get_global_id(0);

    if (t < numberOfRows)
    {
        FPTYPE result = 0.0;
        int max = rowLengths[t];
        for (int i = 0; i < max; i++)
        {
            int ind = t * maxRowLength + i;  // values stored in row major layout
            result += val[ind] * vec[cols[ind]];
        }
        out[t] = result;
    }
}


