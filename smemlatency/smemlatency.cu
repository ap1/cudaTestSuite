
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cutil.h>
#include <ctime>
#include <assert.h>

#define ALLOCATED_SMEM_INDICES  (1024)

// --- kernel --- //

__global__ void testSMemLatency(
        float* random_output_array, long N_SMEM_FETCHES){

    // idea is simple: thread 0 loads one value from smem
    // in a dependent fashion

    __shared__ float sharedVals[ALLOCATED_SMEM_INDICES];

    // create a circular linked list
    // (use all threads)
    int idx = threadIdx.x;
    while(idx < ALLOCATED_SMEM_INDICES){
        sharedVals[idx] = (float)((idx+1)%ALLOCATED_SMEM_INDICES);
        idx += blockDim.x;
    }
    __syncthreads();

    float sum=1.0f;

    // to avoid bank conflicts,
    // only use one thread
    if(threadIdx.x==0){
#pragma unroll 100
        for(long i=0; i<N_SMEM_FETCHES; i++){
            sum = sharedVals[(int)sum];
        }
    }

    __syncthreads();

    if(threadIdx.x==0){
        random_output_array[blockIdx.x] = sum;
    }
}


// --- wrappers --- //

clock_t startTime;

void    startTimer();
float   getTimer();
void    runSMemLatencyTest();


int main(){
    cudaSetDevice(0);
    runSMemLatencyTest();
}

void runSMemLatencyTest(){
    int nBlocks, nThreads;

    nBlocks  = 1;
    nThreads = 32;

    // allocate temp output array
    float* random_output_array;
    cudaMalloc((void**)&random_output_array, nBlocks*sizeof(float));

    // trial run
    testSMemLatency<<<nBlocks, nThreads>>>(random_output_array,100);
    cudaThreadSynchronize(); CUT_CHECK_ERROR("");


    for(int run=0; run<10; run++){

        long baseSmemFetchCount = 500000;
        long sMemFetchCount = baseSmemFetchCount * (run+1);
        int nTests = 100;

        assert(sMemFetchCount % 100 == 0);

        startTimer();
        for(int i=0; i<nTests; i++){
            testSMemLatency<<<nBlocks, nThreads>>>
                (random_output_array, sMemFetchCount);
        }
        cudaThreadSynchronize();
        float timer = getTimer();
        CUT_CHECK_ERROR("");
        printf("%d, %f, // %d tries, total %f ms, %0.2fms per kernel\n", 
                sMemFetchCount, timer/(float)nTests,
                nTests, timer, timer/(float)nTests);
    }

}


// reset clock
void startTimer(){
    startTime = clock();
}

// result is in ms
float getTimer(){
    return (1000.0f*((float)(clock()-startTime))/(float)CLOCKS_PER_SEC);
}
