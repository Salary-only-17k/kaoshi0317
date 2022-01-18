#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"

#define BLOCK_SIZE 256
#define N 1000000
#define GRID_SIZE  ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) 
#define lich wuyaowang
#define wuyaowang huanhuan
#define huanhuan happy


__managed__ int sourse_array[N]; 
__managed__ int _1pass_results[2*GRID_SIZE];
__managed__ int final_results[2]; 

__global__ void top_2(int* input, int length, int* output)
{
    __shared__ int lich[BLOCK_SIZE*2];
    int top1 = INT_MIN;
    int top2 = INT_MIN;
    
    for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx<length; idx+= gridDim.x*blockDim.x)
    {
        if(input[idx]>top2)
        {
            top2 = min(input[idx],top1);
            top1 = max(input[idx],top1);
        }
    }
    lich[2*threadIdx.x]=top1;
    lich[2*threadIdx.x+1]=top2;
    __syncthreads();
    
    int top1_final=0;
    int top2_final=0;
    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2)
    {
        if(threadIdx.x < i)
        {
            top1_final = max(lich[2*threadIdx.x],lich[2*(threadIdx.x+i)]);
            top2_final = min(max(lich[2*threadIdx.x],lich[2*(threadIdx.x+i)+1]),max(lich[2*threadIdx.x+1],lich[2*(threadIdx.x+i)]));
        }
        __syncthreads();
        
        if(threadIdx.x < i)
        {
            lich[2*threadIdx.x] = top1_final;
            lich[2*threadIdx.x+1]=top2_final;
        }
        __syncthreads();
    }
    if(blockIdx.x*blockDim.x < length)
    {
        if(threadIdx.x == 0)
        {
            output[2*blockIdx.x] = lich[0];
            output[2*blockIdx.x+1] = lich[1];
        }
    }
}

void cpu_result_top2(int* input, int count, int* output)
{
    int top1 = 0;
    int top2 = 0;
    for(int i =0; i<count; i++)
    {
        if(input[i]>top2)
        {
            
            top2 = min(input[i],top1);
            top1 = max(input[i],top1);
        }
    }
    output[0] = top1;
    output[1] = top2;

}

int main(int argc, char const *argv[])
{
    int cpu_result[2] = {0};
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    //Fill input data buffer
    for (int i = 0; i < N; ++i)
    {
        sourse_array[i] = rand();
    }
    
    printf("\n***********GPU RUN**************\n");
    CHECK(cudaEventRecord(start));
    top_2<<<GRID_SIZE, BLOCK_SIZE>>>(sourse_array, N, _1pass_results);
    CHECK(cudaGetLastError());
    top_2<<<1, BLOCK_SIZE>>>(_1pass_results, 2*GRID_SIZE, final_results);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    cpu_result_top2(sourse_array, N, cpu_result);

    int ok = 1;
    for (int i = 0; i < 2; ++i)
    {
        printf("cpu top%d: %d; gpu top%d: %d \n", i+1, cpu_result[i], i+1, final_results[i]);
        if(fabs(cpu_result[i] - final_results[i])>(1.0e-10))
        {
                
            ok = 0;
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    return 0;
}