#include<stdio.h>
#include<stdint.h>
#include<time.h>     //for time()
#include<stdlib.h>   //for srand()/rand()
#include<sys/time.h> //for gettimeofday()/struct timeval

#define KEN_CHECK(r) \
{\
    cudaError_t rr = r;   \
    if (rr != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA Error %s, function: %s, line: %d\n",       \
        cudaGetErrorString(rr), __FUNCTION__, __LINE__); \
        exit(-1);\
    }\
}

#define N 100000
#define MAGIC 25000
#define BLOCK_SIZE 256
#define BLOCKS ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) //try next line if you can
//#define BLOCKS 666

typedef struct
{
   int x;
   int y;
}dot;

__managed__ dot source[N];               //input data
__managed__ int final_result = 0;   //scalar output

__global__ void gpu_sum(dot *input, int count, int *output)
{
    //Need your cuda code here
    __shared__ int sum_per_block[BLOCK_SIZE];
    int xx=blockIdx.x * blockDim.x + threadIdx.x;
    dot d = input[xx];
    
    int dx = d.x - 2000;
    int dy = d.y - 2000;
    int radius = 2000;
    if (dx * dx + dy * dy < radius * radius) sum_per_block[threadIdx.x]=1;
    else sum_per_block[threadIdx.x]=0;
    __syncthreads();
    for (int length = BLOCK_SIZE / 2; length >= 1; length /= 2)
    {
        int dd = -1;
        if (threadIdx.x < length)
        {
            dd = sum_per_block[threadIdx.x] + sum_per_block[threadIdx.x + length];
        }
        __syncthreads(); 

        if (threadIdx.x < length)
        {
            sum_per_block[threadIdx.x] = dd;
        }
        __syncthreads();
    }
    
    if (blockDim.x * blockIdx.x < count) 
    {
        if (threadIdx.x == 0) atomicAdd(output, sum_per_block[0]);
    }
}

int cpu_sum(dot *ptr, int count)
{
    int sum = 0;
    for (int i = 0; i < count; i++)
    {
        dot d = ptr[i];
        int dx = d.x - 2000; //circle center: (2000, 2000). radius: 2000
        int dy = d.y - 2000;
        int radius = 2000;
        if (dx * dx + dy * dy < radius * radius) sum ++;
    }
    return sum;
}

void init(dot *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    for (int i = 0; i < count; i++)
    {
        ptr[i].x = rand() % 4001;
        ptr[i].y = rand() % 4001;
    }
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

int main()
{
    //**********************************
    fprintf(stderr, "Preparing the input buffer with %d elements...\n", N);
    init(source, N);

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    //Good luck & have fun!
    fprintf(stderr, "Running on GPU...\n");
    
    double t0 = get_time();
    gpu_sum<<<BLOCKS, BLOCK_SIZE>>>(source, N, &final_result);
    KEN_CHECK(cudaGetLastError());  //checking for launch failures
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
    double t1 = get_time();

    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

    double t2 = get_time();
    int cpu_result = cpu_sum(source, N);
    double t3 = get_time();

    //******The last judgement**********
    if (final_result == cpu_result)
    {
        fprintf(stderr, "Verification Passed!\n");
    }
    else
    {
        fprintf(stderr, "Verification failed! cpu:%d gpu:%d\n",cpu_result,final_result);
        exit(-1);
    }
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    float my_pi = (float)final_result / MAGIC;
    float rel_err = fabsf(my_pi / 3.1415927f - 1.0f);

    fprintf(stderr, "The pi you got is %f, relative error = %f%%\n", my_pi, rel_err * 100.0f);

    if (rel_err < 0.01f)
    {
       printf("passed\n");
    }
    else
    {
       printf("failed. The constant PI is 3.14159265. You know it\n");
    }
    return 0;
}
