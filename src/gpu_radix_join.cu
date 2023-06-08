#include "gpu_radix_join.h"
#include <sched.h>              /* CPU_ZERO, CPU_SET */
#include <pthread.h>            /* pthread_* */
#include <stdlib.h>             /* malloc, posix_memalign */
#include <sys/time.h>           /* gettimeofday */
#include <stdio.h>              /* printf */

#include "prj_params.h"         /* constant parameters */
#include "rdtsc.h"              /* startTimer, stopTimer */
#include "barrier.h"            /* pthread_barrier_* */
#include "generator.h"          /* numa_localize() */

#ifndef BARRIER_ARRIVE
/** barrier wait macro */
#define BARRIER_ARRIVE(B,RV)                            \
    RV = pthread_barrier_wait(B);                       \
    if(RV !=0 && RV != PTHREAD_BARRIER_SERIAL_THREAD){  \
        printf("Couldn't wait on barrier\n");           \
        exit(EXIT_FAILURE);                             \
    }

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)
#endif


/** print out the execution time statistics of the join */
static void 
print_timing(uint64_t total, uint64_t build, uint64_t part,
             uint64_t numtuples, int64_t result,
             struct timeval * start, struct timeval * end)
{
    double diff_usec = (((*end).tv_sec*1000000L + (*end).tv_usec)
                        - ((*start).tv_sec*1000000L+(*start).tv_usec));
    double cyclestuple = total;
    cyclestuple /= numtuples;
    fprintf(stdout, "RUNTIME TOTAL: %lu, BUILD: %lu, PART (cycles): %lu\n", total, build, part);
    fprintf(stdout, "TOTAL-TIME-USECS: %.4lf, TOTAL-TUPLES: %lu, CYCLES-PER-TUPLE: %.4lf\n", diff_usec, result, cyclestuple);
    fflush(stdout);
}
/** 
 * Radix clustering algorithm which does not put padding in between
 * clusters. This is used only by single threaded radix join implementation RJ.
 * 
 * @param outRel 
 * @param inRel 
 * @param hist 
 * @param R 
 * @param D 
 */
void 
radix_partition(relation_t * outRel, relation_t * inRel, int R, int D)
{
    tuple_t ** dst;
    tuple_t * input;
    /* tuple_t ** dst_end; */
    uint32_t * tuples_per_cluster;
    uint32_t i;
    uint32_t offset;
    const uint32_t M = ((1 << D) - 1) << R;
    const uint32_t fanOut = 1 << D;
    const uint32_t ntuples = inRel->num_tuples;

    tuples_per_cluster = (uint32_t*)calloc(fanOut, sizeof(uint32_t));
    /* the following are fixed size when D is same for all the passes,
       and can be re-used from call to call. Allocating in this function 
       just in case D differs from call to call. */
    dst     = (tuple_t**)malloc(sizeof(tuple_t*)*fanOut);
    /* dst_end = (tuple_t**)malloc(sizeof(tuple_t*)*fanOut); */

    input = inRel->tuples;
    /* count tuples per cluster */
    for( i=0; i < ntuples; i++ ){
        uint32_t idx = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
        tuples_per_cluster[idx]++;
        input++;
    }

    offset = 0;
    /* determine the start and end of each cluster depending on the counts. */
    for ( i=0; i < fanOut; i++ ) {
        dst[i]      = outRel->tuples + offset;
        offset     += tuples_per_cluster[i];
        /* dst_end[i]  = outRel->tuples + offset; */
    }

    input = inRel->tuples;
    /* copy tuples to their corresponding clusters at appropriate offsets */
    for( i=0; i < ntuples; i++ ){
        uint32_t idx   = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
        *dst[idx] = *input;
        ++dst[idx];
        input++;
        /* we pre-compute the start and end of each cluster, so the following
           check is unnecessary */
        /* if(++dst[idx] >= dst_end[idx]) */
        /*     REALLOCATE(dst[idx], dst_end[idx]); */
    }

    /* clean up temp */
    /* free(dst_end); */
    free(dst);
    free(tuples_per_cluster);
}

/**
 * \brief dimension of grid is number of fanout
 * dimension of block is number per fanout
*/
__global__ void kernel_gpu_join(tuple_t * R_gpu, tuple_t * S_gpu, int numR, int numS, int* hist_R, int* hist_S, int64_t* matches)
{
    int match = 0;
    int numR_part = numR / (1 << NUM_RADIX_BITS);
    int numS_part = numS / (1 << NUM_RADIX_BITS);
    // R relation of this block
    tuple_t * R_gpu_part = R_gpu + blockIdx.x * numR_part;
    tuple_t * S_gpu_part = S_gpu + blockIdx.x * numS_part;
    uint32_t s = S_gpu_part[threadIdx.x].key;
    for(int offset = 0; offset < numR_part; offset += 32)
    {
        uint32_t r = (uint32_t)R_gpu_part[offset + threadIdx.x].key;
        uint32_t mask = 0xffffffff;
        // magic number 24 represent log2 of number of relation R
        for(int i = NUM_RADIX_BITS; i < 24; i++)
        {
            uint32_t bit = 1 << i;
            uint32_t vote = __ballot(r & bit);
            // uint32_t vote = __ballot_sync(bit, r);
            mask = mask & ((s & bit) ? vote : ~(vote));
        }
        while(mask != 0)
        {
            match++;
            mask &= mask - 1;
        }
        __syncthreads();
    }
    matches[blockIdx.x * blockDim.x + threadIdx.x] = match;
}

/**
 * \brief merge array of length blockDim.x into the first element
 * @param matches array
 * @param blockDim.x length of array
*/
__global__ void kernel_gpu_add(int64_t* matches)
{
    // blockDim.x always= 1024 (for too large array)
    for(int i = (blockDim.x >> 1); i > 0;i >>= 1)
    {
        if(threadIdx.x < i)
        {
            matches[blockIdx.x * blockDim.x + threadIdx.x] += matches[blockIdx.x * blockDim.x + threadIdx.x+i];
        }
        __syncthreads();
    }
    for(int i = (gridDim.x >> 1); i > 0; i >>= 1)
    {
        if(threadIdx.x < i)
        {

        }
    }
}

result_t * gpu_join(relation_t * relR, relation_t * relS, int nthreads)
{
    int64_t result = 0;
    result_t * joinresult;
    uint32_t i;

#ifndef NO_TIMING
    struct timeval start, end;
    uint64_t timer1, timer2, timer3;
#endif

    relation_t *outRelR, *outRelS;

    outRelR = (relation_t*) malloc(sizeof(relation_t));
    outRelS = (relation_t*) malloc(sizeof(relation_t));

    joinresult = (result_t *) malloc(sizeof(result_t));

    /* allocate temporary space for partitioning */
    size_t sz = relR->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
    outRelR->tuples     = (tuple_t*) malloc(sz);
    outRelR->num_tuples = relR->num_tuples;

    sz = relS->num_tuples * sizeof(tuple_t) + RELATION_PADDING;
    outRelS->tuples     = (tuple_t*) malloc(sz);
    outRelS->num_tuples = relS->num_tuples;

#ifndef NO_TIMING
    gettimeofday(&start, NULL);
    startTimer(&timer1);
    startTimer(&timer2);
    startTimer(&timer3);
#endif

    /* apply radix-clustering on relation R for pass-1 */
    radix_partition(outRelR, relR, 0, NUM_RADIX_BITS);
    relR = outRelR;

    /* apply radix-clustering on relation S for pass-1 */
    radix_partition(outRelS, relS, 0, NUM_RADIX_BITS);
    relS = outRelS;


#ifndef NO_TIMING
    stopTimer(&timer3);
#endif

    int * R_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));
    int * S_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));

    /* compute number of tuples per cluster */
    for( i=0; i < relR->num_tuples; i++ ){
        uint32_t idx = (relR->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
        R_count_per_cluster[idx] ++;
    }
    for( i=0; i < relS->num_tuples; i++ ){
        uint32_t idx = (relS->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
        S_count_per_cluster[idx] ++;
    }

    tuple_t * R_gpu;
    tuple_t * S_gpu;
    cudaMalloc((void**)&R_gpu, relR->num_tuples * sizeof(tuple_t));
    cudaMalloc((void**)&S_gpu, relS->num_tuples * sizeof(tuple_t));
    cudaMemcpy(R_gpu, relR->tuples, relR->num_tuples * sizeof(tuple_t), cudaMemcpyHostToDevice);
    cudaMemcpy(S_gpu, relS->tuples, relS->num_tuples * sizeof(tuple_t), cudaMemcpyHostToDevice);
    int * hist_R;
    int * hist_S;
    cudaMalloc((void**)&hist_R, (1<<NUM_RADIX_BITS) * sizeof(int));
    cudaMalloc((void**)&hist_S, (1<<NUM_RADIX_BITS) * sizeof(int));
    cudaMemcpy(hist_R, R_count_per_cluster, (1<<NUM_RADIX_BITS) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hist_S, S_count_per_cluster, (1<<NUM_RADIX_BITS) * sizeof(int), cudaMemcpyHostToDevice);

    // record number of matched number
    int64_t * matches;
    cudaMalloc((void**)&matches, (relR->num_tuples) * sizeof(int64_t));

    dim3 threaddim(relR->num_tuples / (1<<NUM_RADIX_BITS));
    dim3 blockdim((1<<NUM_RADIX_BITS));

    kernel_gpu_join<<<blockdim, threaddim>>>(R_gpu, S_gpu, relR->num_tuples, relS->num_tuples, hist_R, hist_S, matches);

    dim3 add_griddim(1);
    dim3 add_blockdim(relR->num_tuples);
    // kernel_gpu_add<<<add_griddim, add_blockdim>>>(matches);
    int64_t* matches_cpu = (int64_t*)malloc((relR->num_tuples) * sizeof(int64_t));
        cudaMemcpy(matches_cpu, matches, (relR->num_tuples) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < relR->num_tuples; i++)
    {
        result += matches_cpu[i];
    }
    // cudaMemcpy(&result, matches, sizeof(int64_t), cudaMemcpyDeviceToHost);

#ifndef NO_TIMING
    /* TODO: actually we're not timing build */
    stopTimer(&timer2);/* build finished */
    stopTimer(&timer1);/* probe finished */
    gettimeofday(&end, NULL);
    /* now print the timing results: */
    print_timing(timer1, timer2, timer3, relS->num_tuples, result, &start, &end);
#endif

    /* clean-up temporary buffers */
    free(S_count_per_cluster);
    free(R_count_per_cluster);

#if NUM_PASSES == 1
    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);
#endif

    joinresult->totalresults = result;
    joinresult->nthreads     = 1;

    return joinresult;
}