#ifdef VTYPE_FLOAT
#define VTYPE float
#define VTYPE_MAX 3.40282347e+38
#else
#ifdef VTYPE_DOUBLE_KHR
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define VTYPE double
#elif VTYPE_DOUBLE_AMD
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define VTYPE double
#endif
#define VTYPE_MAX 1.79769313e+308
#endif

__kernel void
kmeans_kernel_col(__global VTYPE  *feature,   
              __global VTYPE  *clusters,
              __global int    *membership,
                int     npoints,
                int     nclusters,
                int     nfeatures
              ) 
{
    unsigned int point_id = get_global_id(0);
    int index = 0;
    if (point_id < npoints)
    {
        VTYPE min_dist = VTYPE_MAX;
        for (int i=0; i < nclusters; i++) {
            
            VTYPE dist = 0;
            VTYPE ans  = 0;
            for (int l=0; l<nfeatures; l++){
                    ans += (feature[l * npoints + point_id]-clusters[i*nfeatures+l])* 
                           (feature[l * npoints + point_id]-clusters[i*nfeatures+l]);
            }

            dist = ans;
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
                
            }
        }
        membership[point_id] = index;
    }
}

__kernel void
kmeans_kernel_row(__global VTYPE  *feature,   
              __global VTYPE  *clusters,
              __global int    *membership,
                int     npoints,
                int     nclusters,
                int     nfeatures
              ) 
{
    unsigned int point_id = get_global_id(0);
    int index = 0;
    if (point_id < npoints)
    {
        VTYPE min_dist = VTYPE_MAX;
        for (int i=0; i < nclusters; i++) {
            
            VTYPE dist = 0;
            VTYPE ans  = 0;
            for (int l=0; l<nfeatures; l++){
                    ans += (feature[point_id * nfeatures + l]-clusters[i*nfeatures+l])* 
                           (feature[point_id * nfeatures + l]-clusters[i*nfeatures+l]);
            }

            dist = ans;
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
                
            }
        }
        membership[point_id] = index;
    }
}

__kernel void
kmeans_transpose(__global VTYPE  *feature, __global VTYPE  *feature_swap,
            int npoints, int nfeatures){
    unsigned int tid = get_global_id(0);
    if (tid >= npoints)
        return;

    for(int i = 0; i <  nfeatures; i++) {
        feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
    }
}
