#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"

#define N 1000
#define BLOCK_SIZE 32

__managed__ int input_Matrix[N][N];
__managed__ int output_GPU[N][N];
__managed__ int output_CPU[N][N];
__global__ void gpu_kernel(int input_M[N][N], int output_M[N][N])
{
    int row=blockIdx.y*blockDim.y + threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col < N && row <N){
        if(col%2==0 and row%2==0){
            output_M[col][row] = input_M[col][row]*input_M[col][row];
        }
        else{
            output_M[col][row] = input_M[col][row]-1;
        }
    }
}
void cpu_kernel(int intput_M[N][N], int output_CPU[N][N])
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(j%2==0 && i%2==0)
            {
                output_CPU[i][j] = intput_M[i][j]*intput_M[i][j];
            }
            else
            {
                output_CPU[i][j] = intput_M[i][j]-1;
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    
    cudaEvent_t start,stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_gpu));


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) 
        {
            
            input_Matrix[i][j] = rand()%3001;
            //printf("%d ",input_Matrix[i][j]);
        }
        //printf("\n");
    }
    cpu_kernel(input_Matrix, output_CPU);
    
    CHECK(cudaEventRecord(start));
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("\n***********GPU RUN**************\n");
    gpu_kernel<<<dimGrid, dimBlock>>>(input_Matrix, output_GPU);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    
    float elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_gpu));

    int ok = 1;
    for (int i = 0; i < N; ++i)
    { 
        for (int j = 0; j < N; ++j)
        {
            //printf("%d ",output_GPU[i][j]);
            if(fabs(output_GPU[i][j] - output_CPU[i][j])>(1.0e-10))
            {
                ok = 0;
            }
            
        }
        //printf("\n");
    }
    printf("\n***********Check result**************\n");
    for (int i = 0; i < N; ++i)
    { 
        for (int j = 0; j < N; ++j)
        {
            //printf("%d ",output_CPU[i][j]);
            if(fabs(output_GPU[i][j] - output_CPU[i][j])>(1.0e-10))
            {
                ok = 0;
            }
            
        }
        //printf("\n");
    }


    if(ok)
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