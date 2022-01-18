#include <stdio.h>
#include <stdlib.h>
#include <time.h>   
#include "error.cuh"

#define BLOCK_SIZE 256
#define N 1000000
#define GRID_SIZE  ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) 
#define topk 10


__managed__ int source_array[N];
__managed__ int _1pass_results[topk * GRID_SIZE];
__managed__ int final_results[topk];

__device__ __host__ void insert_value(int* array, int k, int data)
{
    for (int i = 0; i < k; i++)
    {
        if (array[i] == data)
        {
            return;
        }
    }
    if (data < array[k - 1])
        return;
    for (int i = k - 2; i >= 0; i--)
    {
        if (data > array[i])
            array[i + 1] = array[i];
        else {
            array[i + 1] = data;
            return;
        }
    }
    array[0] = data;
}

__global__ void top_k(int* input, int length, int* output, int k)
{
    __shared__ int lich[BLOCK_SIZE * topk];
    int top_array[topk];

    for (int i = 0; i < topk; i++)
    {
        top_array[i] = INT_MIN;
    }

    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x)
    {
        insert_value(top_array, k, input[idx]);
    }
//#pragma unroll 5
    for (int j = 0; j < topk; j++)
    {
        lich[topk * threadIdx.x+j] = top_array[j];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE / 2; i >= 1; i /= 2)
    {
        if (threadIdx.x < i)
        {
            for (int m = 0; m < topk; m++)
            {
                insert_value(top_array, topk, lich[topk * (threadIdx.x + i) + m]);
            }
        }
        __syncthreads();

        if (threadIdx.x < i)
        {
//#pragma unroll 5
            for (int m = 0; m < topk; m++)
            {
                lich[topk* threadIdx.x + m] = top_array[m];
            }
        }
        __syncthreads();
    }
    if (blockIdx.x * blockDim.x < length)
    {
        if (threadIdx.x == 0)
        {
//#pragma unroll 5
            for (int m = 0; m < topk; m++)
            {
                output[topk * blockIdx.x + m] = lich[m];
            }
        }
    }
}

void cpu_result_topk(int* input, int count, int* output)
{
    /*for (int i = 0; i < topk; i++)
    {
        output[i] = INT_MIN;
    }*/
    for (int i = 0; i < count; i++)
    {
        insert_value(output, topk, input[i]);

    }
}

void _init(int* ptr, int count)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

int main(int argc, char const* argv[])
{
    int cpu_result[topk] = { 0 };
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    //Fill input data buffer
    _init(source_array, N);


    printf("\n***********GPU RUN**************\n");
    CHECK(cudaEventRecord(start));
    top_k << <GRID_SIZE, BLOCK_SIZE >> > (source_array, N, _1pass_results, topk);
    CHECK(cudaGetLastError());
    top_k << <1, BLOCK_SIZE >> > (_1pass_results, topk * GRID_SIZE, final_results, topk);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    cpu_result_topk(source_array, N, cpu_result);

    int ok = 1;
    for (int i = 0; i < topk; ++i)
    {
        printf("cpu top%d: %d; gpu top%d: %d \n", i + 1, cpu_result[i], i + 1, final_results[i]);
        if (fabs(cpu_result[i] - final_results[i]) > (1.0e-10))
        {

            ok = 0;
        }
    }

    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    return 0;
}