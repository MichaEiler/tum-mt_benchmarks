#ifdef VTYPE_FLOAT
#define VTYPE float
#elif VTYPE_DOUBLE_KHR
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VTYPE double
#elif VTYPE_DOUBLE_AMD
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VTYPE double
#else
#error No precision defined.
#endif


inline int ToGlobalRow( int gidRow, int lszRow, int lidRow ) {
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global row (without halo)
    return gidRow*lszRow + lidRow;
}

inline int ToGlobalCol( int gidCol, int lszCol, int lidCol ) {
    // assumes coordinates and dimensions are logical (without halo)
    // returns logical global column (without halo)
    return gidCol*lszCol + lidCol;
}


inline int ToFlatHaloedIdx( int row, int col, int rowPitch ) {
    // assumes input coordinates and dimensions are logical (without halo)
    // and a halo of width 1
    return (row + 1)*(rowPitch + 2) + (col + 1);
}

__kernel void StencilKernel( __global VTYPE* data, __global VTYPE* newData, const int alignment,
                VTYPE wCenter, VTYPE wCardinal, VTYPE wDiagonal) {
    __local VTYPE sh[(LOCAL_ROWS + 2) * (LOCAL_COLUMNS + 2)];

    if ((get_global_id(1) >= (GLOBAL_COLUMNS - 2)) || (get_global_id(0) >= ((GLOBAL_ROWS - 2)/LOCAL_ROWS))) {
        return;
    }

    // determine our location in the OpenCL coordinate system
    // To match with the row-major ordering used to store the 2D
    // array in both the host and on the device, we use:
    //   dimension 0 == rows,
    //   dimension 1 == columns
    int gidRow = get_group_id(0);
    int gidCol = get_group_id(1);
    int gszRow = get_num_groups(0);
    int gszCol = get_num_groups(1);
    int lidRow = get_local_id(0);
    int lidCol = get_local_id(1);
    int lszRow = LOCAL_ROWS;
    int lszCol = get_local_size(1);

    // determine our logical global data coordinates (without halo)
    int gRow = ToGlobalRow( gidRow, lszRow, lidRow );
    int gCol = ToGlobalCol( gidCol, lszCol, lidCol );

    // determine pitch of rows (without halo)
    int nCols = gszCol * lszCol + 2;     // num columns including halo
    int nPaddedCols = nCols;// + (((nCols % alignment) == 0) ? 0 : (alignment - (nCols % alignment)));
    int gRowWidth = nPaddedCols - 2;    // remove the halo

    // Copy my global data item to a shared local buffer.
    // That local buffer is passed to us as a parameter.
    // We assume it is large enough to hold all the data computed by
    // our block, plus a halo of width 1.
    int lRowWidth = lszCol;          // logical, not haloed
    for( int i = 0; i < (lszRow + 2); i++ )
    {
        int lidx = ToFlatHaloedIdx( lidRow - 1 + i, lidCol, lRowWidth );
        int gidx = ToFlatHaloedIdx( gRow - 1 + i, gCol, gRowWidth );
        sh[lidx] = data[gidx];
    }

    // Copy the "left" and "right" halo rows into our local memory buffer.
    // Only two threads are involved (first column and last column).
    if( lidCol == 0 )
    {
        for( int i = 0; i < (lszRow + 2); i++ )
        {
            int lidx = ToFlatHaloedIdx(lidRow - 1 + i, lidCol - 1, lRowWidth );
            int gidx = ToFlatHaloedIdx(gRow - 1 + i, gCol - 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    }
    else if( lidCol == (lszCol - 1) )
    {
        for( int i = 0; i < (lszRow + 2); i++ )
        {
            int lidx = ToFlatHaloedIdx(lidRow - 1 + i, lidCol + 1, lRowWidth );
            int gidx = ToFlatHaloedIdx(gRow - 1 + i, gCol + 1, gRowWidth );
            sh[lidx] = data[gidx];
        }
    }

    // let all those loads finish
    barrier( CLK_LOCAL_MEM_FENCE );

    // do my part of the smoothing operation
    for( int i = 0; i < lszRow; i++ )
    {
        int cidx  = ToFlatHaloedIdx( lidRow     + i, lidCol    , lRowWidth );
        int nidx  = ToFlatHaloedIdx( lidRow - 1 + i, lidCol    , lRowWidth );
        int sidx  = ToFlatHaloedIdx( lidRow + 1 + i, lidCol    , lRowWidth );
        int eidx  = ToFlatHaloedIdx( lidRow     + i, lidCol + 1, lRowWidth );
        int widx  = ToFlatHaloedIdx( lidRow     + i, lidCol - 1, lRowWidth );
        int neidx = ToFlatHaloedIdx( lidRow - 1 + i, lidCol + 1, lRowWidth );
        int seidx = ToFlatHaloedIdx( lidRow + 1 + i, lidCol + 1, lRowWidth );
        int nwidx = ToFlatHaloedIdx( lidRow - 1 + i, lidCol - 1, lRowWidth );
        int swidx = ToFlatHaloedIdx( lidRow + 1 + i, lidCol - 1, lRowWidth );

        VTYPE centerValue = sh[cidx];
        VTYPE cardinalValueSum = sh[nidx] + sh[sidx] + sh[eidx] + sh[widx];
        VTYPE diagonalValueSum = sh[neidx] + sh[seidx] + sh[nwidx] + sh[swidx];

        newData[ToFlatHaloedIdx(gRow + i, gCol, gRowWidth)] = wCenter * centerValue +
                wCardinal * cardinalValueSum +
                wDiagonal * diagonalValueSum;
    }
}

