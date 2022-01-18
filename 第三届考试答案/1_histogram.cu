#include<stdio.h>
#include<stdint.h>
#include<time.h>     //for time()
#include<stdlib.h>   //for srand()/rand()
#include<sys/time.h> //for gettimeofday()/struct timeval
#include<assert.h>

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


#define N 10000000 //~40MB for 4B types
#define FILENAME "zhangzha.dat"
	
//data range: [0, 1000).
/* bucket 0: [0,   50)
   bucket 1: [50,  100)
   bucket 2: [100, 150)
   bucket N: [50 * N, 50 * N + 50)
*/
#define BUCKETS_COUNT 20 

#define BLOCK_SIZE 256
#define BLOCKS ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) 



__global__ void gpu_histogram(int *input, int count, int *output)
{
    __shared__ int partial_result[BUCKETS_COUNT];

    //initialization stage
    if (threadIdx.x < BUCKETS_COUNT)
    {
        partial_result[threadIdx.x] = 0;
    }
    __syncthreads();

    //shared memory stage
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 idx < count;
         idx += blockDim.x * gridDim.x
        )
    {
	int target = input[idx] / 50;
	//if (target >= BUCKETS_COUNT) printf("%d\n", target);
	atomicAdd(&partial_result[target], 1);
    }
    __syncthreads();
 
    //global memory stage
    if (threadIdx.x < BUCKETS_COUNT)
    {
        int count = partial_result[threadIdx.x];
        atomicAdd(&output[threadIdx.x], count);
    }
}

void cpu_histogram(int *input, int count, int *output)
{
    for (int i = 0; i < BUCKETS_COUNT; i++) output[i] = 0;

    for (int i = 0; i < count; i++)
    {
	int target = input[i] / 50;
	assert(target >= 0 && target < BUCKETS_COUNT);
	
	output[target]++;
    }
}


void show(int *buckets)
{
    int max_value = 0;
    for (int i = 0; i < BUCKETS_COUNT; i++) max_value = max(max_value, buckets[i]);

    //scale to 1 - 40 stars
    float scale = 40.0f / (float)max_value;

    for (int i = 0; i < BUCKETS_COUNT; i++)
    {
        printf("[%04d, %04d): ", i * 50, i * 50 + 50);
        int stars = (int)(scale * (float)buckets[i]);
        for (int j = 0; j < 40; j++)
        {
            if (j < stars) printf("*"); else printf(" ");
        }
	printf(" %d\n", buckets[i]);
    }
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

__managed__ int GPU_result[BUCKETS_COUNT];

int main()
{
    int *input_buffer = NULL;
    KEN_CHECK(cudaMallocManaged(&input_buffer, sizeof(int) * N));

    //**********************************
    printf("Loading from file '%s'\n", FILENAME);

    FILE *fp = fopen(FILENAME, "rb");
    assert(fp != NULL);
	
    int items_read = fread(input_buffer, sizeof(int), N, fp);
    assert(items_read == N);

    KEN_CHECK(cudaMemset(GPU_result, 0, sizeof(int) * BUCKETS_COUNT));

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    //Good luck & have fun!
    
    printf("Running on GPU...\n");
    

double t0 = get_time();
    gpu_histogram<<<BLOCKS, BLOCK_SIZE>>>(input_buffer, N, GPU_result);
        KEN_CHECK(cudaGetLastError());  //checking for launch failures
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
double t1 = get_time();


    //*********************************
    printf("GPU histogram goes below\n");
    show(GPU_result);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");


double t2 = get_time();
     int CPU_result[BUCKETS_COUNT];
     cpu_histogram(input_buffer, N, CPU_result);
double t3 = get_time();

    //******The last judgement**********
    int pass = 1;
    for (int i = 0; i < BUCKETS_COUNT; i++) 
    {
        if (GPU_result[i] != CPU_result[i]) pass = 0;
    }
    if (pass)
    {
        fprintf(stderr, "Test Passed!\n");
    }
    else
    {
        fprintf(stderr, "Test failed!\n");
	exit(-1);
    }
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    //select the line you want
    printf("\nFrom the histogram shown above. I think it's uniform distribution\n\n");
    //printf("\nFrom the histogram shown above. I think it's normal distribution\n\n");
	
    return 0;
}	
	
