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

#define POINT_DIMENSION 3

/* ============================================================
//--cambine: kernel funtion of pgain
//--author: created by Jianbin Fang
//--date:   02/03/2011
============================================================ */

typedef struct {
    VTYPE weight;
    int assign;  /* number of point where this one is assigned */
    VTYPE cost;  /* cost of that assignment, weight*distance */
} Point;

__kernel void pgain_kernel( __global Point *p, __global VTYPE *coord_d,
                __global VTYPE * work_mem_d, __global int *center_table_d,
                __global char *switch_membership_d, int num, long x, int K) {
    __local VTYPE coord_s[POINT_DIMENSION];

    /* block ID and global thread ID */
    const int thread_id = get_global_id(0);
    const int local_id = get_local_id(0);

    if(thread_id >= num)
        return;

     // coordinate mapping of point[x] to shared mem
    if(local_id == 0) {
        coord_s[0] = coord_d[x];
        coord_s[1] = coord_d[num + x];
        coord_s[2] = coord_d[num + num + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
   
    // cost between this point and point[x]: euclidean distance multiplied by weight
    VTYPE x_cost = 0.0;
    for (int i=0; i<POINT_DIMENSION; i++)
        x_cost += (coord_d[(i*num)+thread_id]-coord_s[i]) * (coord_d[(i*num)+thread_id]-coord_s[i]);
    x_cost = x_cost * p[thread_id].weight;
   
    VTYPE current_cost = p[thread_id].cost;

    int base = thread_id*(K+1);
    // if computed cost is less then original (it saves), mark it as to reassign
    if ( x_cost < current_cost ){
        switch_membership_d[thread_id] = '1';
        int addr_1 = base + K;
        work_mem_d[addr_1] = x_cost - current_cost;
    }
    // if computed cost is larger, save the difference
    else {
        int assign = p[thread_id].assign;
        int addr_2 = base + center_table_d[assign];
        work_mem_d[addr_2] += current_cost - x_cost;
    }
}

