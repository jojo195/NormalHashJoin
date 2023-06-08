/**
 * @file    parallel_radix_join.h
 * @author Tengyuan Jin from Nankai University NBJL
 * @date    2023.6.8
 * @version GPU accelerated hash join
 * 
 * @brief  GPU accelerated partitioning hash join
 * 
 * (c) 2023 Nankai University NBJL
 * 
 */

#ifndef __GPU_RADIX_JOIN__
#define __GPU_RADIX_JOIN__

#include "types.h" /* relation_t */

result_t *
gpu_join(relation_t * relR, relation_t * relS, int nthreads);

#endif