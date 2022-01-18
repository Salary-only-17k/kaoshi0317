#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"

#define N 1000
#define BLOCK_SIZE 256

__managed__ int input_Matrix[N*N];
__managed__ int output_GPU[N*N];
__managed__ int output_CPU[N*N];

__global__ void gpu_kernel(int input_M[N*N], int output_M[N*N])
{
    int row=blockIdx.y*blockDim.y + threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col < N && row <N){
        if(col%2==0 and row%2==0){
            
        }
    }

}

void cpu_kernel(int input_M[N*N], int output_CPU[N*N])
{
    for (int i = 0; i < N*N; i++)
    {
        if (input_M[i]<=100)
        {
            output_CPU[i] = 0;
        }
        else
        {
            output_CPU[i] = 1;
        }
    }
}

int main(int argc, char const* argv[])
{
    cudaEvent_t start, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_gpu));

    for (int i = 0; i < N*N; ++i) {
        input_Matrix[i] = rand() % 200;
        //printf("%d ",input_Matrix[i][j]);
    }
    cpu_kernel(input_Matrix, output_CPU);

    CHECK(cudaEventRecord(start));
    /*unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    */
    unsigned int grid_size = (N*N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("\n***********GPU RUN**************\n");
    gpu_kernel <<<grid_size, BLOCK_SIZE >>> (input_Matrix, output_GPU);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

    float elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_gpu));

    int ok = 1;
    printf("\n***********Check result**************\n");
    for (int i = 0; i < N*N; ++i)
    {
        //printf("%d ",output_GPU[i][j]);
        if (fabs(output_GPU[i] - output_CPU[i]) > (1.0e-10))
        {
            ok = 0;
            //printf("[%d:%d] ", i, output_CPU[i]);
        }
        //printf("\n");
    }

    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    return 0;
}